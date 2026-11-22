/**
 * @file test_gnc.c
 * @brief Guidance and control unit tests.
 */
#include <math.h>
#include <stdio.h>

#include "aero_model.h"
#include "control.h"
#include "fsw_config.h"
#include "gnc_math.h"
#include "guidance.h"
#include "test_framework.h"

/* ---------------- guidance ---------------- */
static void test_pdg_tgo_is_global_minimum(void)
{
    const float r[3] = { -300, 120, 650 }, v[3] = { 25, -6, -70 };
    const float rf[3] = { 0, 0, 30 }, vf[3] = { 0, 0, -3 }, a[3] = { 0, 0, -FSW_G0 + 1.5f };
    const float gam = 8.0f;
    float t_opt = pdg_optimal_tgo(r, v, rf, vf, a, gam, 1.0f, 90.0f);
    float best = 1e30f, tb = 0.0f, T;
    for (T = 1.0f; T <= 90.0f; T += 0.01f) {
        const float J = pdg_cost(r, v, rf, vf, a, gam, T);
        if (J < best) { best = J; tb = T; }
    }
    printf("    tgo* = %.3f s (brute force %.3f s)\n", t_opt, tb);
    CHECK_NEAR(t_opt, tb, 0.05);
}

static void test_pdg_closed_loop_point_mass(void)
{
    /* unconstrained ZEM/ZEV on a point mass with drift must hit the gate state */
    pdg_t g;
    float r[3] = { -300, 120, 650 }, v[3] = { 25, -6, -70 };
    const float rf[3] = { 0, 0, 30 }, vf[3] = { 0, 0, -3 }, a[3] = { 0, 0, -FSW_G0 };
    const float dt = 0.01f;
    int k, i;
    float t = 0.0f, energy = 0.0f;
    pdg_init(&g, 8.0f, rf, vf);
    for (k = 0; k < 20000; ++k) {
        pdg_step(&g, r, v, a, dt);
        if (g.tgo <= 0.31f) { break; }
        for (i = 0; i < 3; ++i) {
            const float acc = g.a_cmd[i] + a[i];
            r[i] += v[i] * dt + 0.5f * acc * dt * dt;
            v[i] += acc * dt;
        }
        energy += v3_dot(g.a_cmd, g.a_cmd) * dt;
        t += dt;
    }
    /* finish the last 0.3 s with the final command held (gains are singular at tgo -> 0) */
    {
        const float Tl = g.tgo;
        for (i = 0; i < 3; ++i) {
            const float acc = g.a_cmd[i] + a[i];
            r[i] += v[i] * Tl + 0.5f * acc * Tl * Tl;
            v[i] += acc * Tl;
        }
        t += Tl;
    }
    printf("    reached gate at t=%.2f s: r=(%.2f %.2f %.2f) v=(%.2f %.2f %.2f)\n",
           t, r[0], r[1], r[2], v[0], v[1], v[2]);
    for (i = 0; i < 3; ++i) {
        CHECK_NEAR(r[i], rf[i], 0.5);
        CHECK_NEAR(v[i], vf[i], 0.5);
    }
    (void)energy;
}

static void test_thrust_projection(void)
{
    float F[3];
    const float tilt = 20.0f * GNC_DEG2RAD;
    /* inside: untouched */
    float in[3] = { 1e5f, 0, 2e6f };
    CHECK(thrust_project(in, 6e5f, 3.2e6f, tilt, F) == 0);
    CHECK_NEAR(F[0], 1e5, 1.0); CHECK_NEAR(F[2], 2e6, 1.0);
    /* too tilted: horizontal clipped to the cone, vertical preserved */
    { float d[3] = { 2e6f, 0, 1e6f };
      CHECK(thrust_project(d, 6e5f, 3.2e6f, tilt, F) == 1);
      CHECK_NEAR(F[2], 1e6, 1.0); CHECK_NEAR(atan2f(F[0], F[2]), tilt, 1e-4); }
    /* too large: vertical priority */
    { float d[3] = { 5e5f, 5e5f, 4e6f };
      thrust_project(d, 6e5f, 3.2e6f, tilt, F);
      CHECK_NEAR(F[2], 3.2e6, 1.0); CHECK_NEAR(F[0], 0.0, 1.0); }
    /* too small: raised to T_min */
    { float d[3] = { 0, 0, 1e5f };
      thrust_project(d, 6e5f, 3.2e6f, tilt, F);
      CHECK_NEAR(v3_norm(F), 6e5, 1.0); }
    /* downward request: minimum thrust straight up */
    { float d[3] = { 0, 1e5f, -1e6f };
      thrust_project(d, 6e5f, 3.2e6f, tilt, F);
      CHECK(F[2] > 0.0f); CHECK_NEAR(v3_norm(F), 6e5, 1.0); }
}

static void test_glide_axis_inverts_crossflow(void)
{
    /* the attitude produced by the inversion must create an aero force along n_des */
    float rho, a;
    int trial;
    atmo_us76(4000.0f, &rho, &a);
    for (trial = 0; trial < 6; ++trial) {
        const float ang = 0.9f * (float)trial;
        const float tilt = 0.05f + 0.05f * (float)trial;
        float n[3] = { sinf(tilt) * cosf(ang), sinf(tilt) * sinf(ang), cosf(tilt) };
        float vair[3] = { 6.0f * cosf(1.3f * ang), -4.0f, -80.0f };
        float xcur[3] = { 1, 0, 0 }, xd[3], q[4], vb[3], F[3], FL[3], flaps[4] = { 0, 0, 0, 0 };
        glide_axis_for_force(vair, n, xcur, xd);
        CHECK(att_from_two_axes(q, 0, 1.0f, xd, 2, vair) == 0);
        q_rotate_inv(vb, q, vair);
        (void)aero_forces(vb, (const float[3]){ 0, 0, 0 }, rho, a, flaps, 1.0f, F, 0);
        q_rotate(FL, q, F);
        (void)v3_normalize(FL, FL);
        /* axial drag (C_A small) tilts the resultant slightly: allow 1.5 deg */
        CHECK(v3_dot(FL, n) > cosf(1.5f * GNC_DEG2RAD));
    }
}

static void test_ignition_altitude_engine_aware(void)
{
    const float h2 = ignition_altitude(-65.0f, 115000.0f, 2, 30.0f);
    const float h1 = ignition_altitude(-65.0f, 115000.0f, 1, 30.0f);
    printf("    ignition altitude at 65 m/s: 2 engines %.0f m, 1 engine %.0f m\n", h2, h1);
    CHECK(h2 > 300.0f && h2 < 900.0f);
    CHECK(h1 > 2.0f * h2);
}

static void test_stopping_margin(void)
{
    CHECK(stopping_margin(500.0f, -60.0f, 110000.0f, 2, 30.0f, 0.0f) < 0.5f);
    CHECK(stopping_margin(150.0f, -60.0f, 110000.0f, 1, 30.0f, 0.0f) > 1.0f);
    CHECK(stopping_margin(150.0f, 1.0f, 110000.0f, 1, 30.0f, 0.0f) == 0.0f);
}

static void test_terminal_descent_profile(void)
{
    const float pad[3] = { 0, 0, 0 }, ag[3] = { 0, 0, -FSW_G0 };
    float r[3] = { 3, -2, 30 }, v[3] = { 0, 0, -3 }, a[3];
    int k, i;
    for (k = 0; k < 6000 && r[2] > 0.0f; ++k) {
        terminal_accel(r, v, pad, 30.0f, 3.0f, ag, 1, a);
        for (i = 0; i < 3; ++i) { v[i] += (a[i] + ag[i]) * 0.01f; r[i] += v[i] * 0.01f; }
    }
    printf("    touchdown v=(%.2f %.2f %.2f) at (%.2f %.2f)\n", v[0], v[1], v[2], r[0], r[1]);
    CHECK_NEAR(v[2], -TERM_V_TD, 0.15);
    CHECK(sqrtf(r[0] * r[0] + r[1] * r[1]) < 0.5f);
}

/* ---------------- control ---------------- */
static void test_gimbal_allocation_signs(void)
{
    alloc_t out;
    const float T = 2.0e6f, off[3] = { 0, 0, 0 }, vb[3] = { -50, 0, 0 };
    const float tau[3] = { 0, 3.0e6f, -2.0e6f };
    float F[3], M[3];
    alloc_torque(tau, T, off, vb, 1.2f, 0.3f, 0, &out);
    /* reproduce the truth-model torque r x F at the engine plane */
    F[0] = T; F[1] = T * out.gimbal[0]; F[2] = T * out.gimbal[1];
    M[1] = -VEH_X_AFT * F[2];
    M[2] = VEH_X_AFT * F[1];
    CHECK_NEAR(M[1], tau[1], 1.0);
    CHECK_NEAR(M[2], tau[2], 1.0);
    CHECK(out.saturated == 0);
}

static void test_engine_out_offset_cancelled(void)
{
    alloc_t out;
    const float T_each[3] = { 1.2e6f, 0.0f, 1.2e6f }, vb[3] = { -50, 0, 0 }, zero[3] = { 0, 0, 0 };
    float off[3];
    engine_offset_torque(T_each, off);
    alloc_torque(zero, 2.4e6f, off, vb, 1.2f, 0.3f, 0, &out);
    CHECK(v3_norm(off) > 1.0e6f);
    CHECK_NEAR(out.tau_ach[1], 0.0, 1.0);
    CHECK_NEAR(out.tau_ach[2], 0.0, 1.0);
}

static void test_flap_allocation_trim_and_bounds(void)
{
    /* belly-flop: request the moment that trims the body's CoP offset */
    float rho, a, vb[3] = { 0, 0, 75 }, F[3], Mb[3], tau[3];
    const float zero_flaps[4] = { 0, 0, 0, 0 }, w0[3] = { 0, 0, 0 };
    alloc_t out;
    const float off[3] = { 0, 0, 0 };
    atmo_us76(5000.0f, &rho, &a);
    (void)aero_forces(vb, w0, rho, a, zero_flaps, 1.0f, F, Mb);
    v3_scale(tau, Mb, -1.0f);
    tau[0] = 2.0e5f;                       /* plus some roll */
    alloc_torque(tau, 0.0f, off, vb, rho, 0.5f, 1, &out);
    printf("    trim flaps = [%.2f %.2f %.2f %.2f]  rcs = [%.0f %.0f %.0f]\n",
           out.flaps[0], out.flaps[1], out.flaps[2], out.flaps[3], out.rcs[0], out.rcs[1], out.rcs[2]);
    CHECK(out.flaps[2] > out.flaps[0]);    /* aft flaps carry the nose-down trim */
    CHECK_NEAR(out.tau_ach[1], tau[1], 0.02 * fabs(tau[1]));
    CHECK_NEAR(out.tau_ach[0], tau[0], 1e3);
    /* infeasible request saturates cleanly */
    tau[1] = -5.0e7f;
    alloc_torque(tau, 0.0f, off, vb, rho, 0.5f, 1, &out);
    CHECK(out.saturated == 1);
    CHECK(out.flaps[0] >= 0.0f && out.flaps[2] <= 1.0f);
}

static void test_attitude_slew_closed_loop(void)
{
    /* 90 deg pitch flip of the rigid body with gimbal authority (rigid-body Euler integration) */
    att_ctrl_t c;
    att_gains_t g = { 1.2f, 0.45f, 0.30f, 4.0f, 0.5f, 0.05f };
    float q[4], qd[4], w[3] = { 0, 0, 0 };
    const float I[3] = { VEH_IXX, VEH_IYY, VEH_IZZ }, off[3] = { 0, 0, 0 }, vb[3] = { -60, 0, 0 };
    float t = 0.0f, max_over = 0.0f, th_final;
    int k, i;
    q_from_euler321(q, 0.0f, 0.0f, GNC_PI);           /* belly down, nose +x */
    q_from_euler321(qd, 0.0f, -GNC_PI / 2.0f, 0.0f);  /* nose up */
    att_ctrl_reset(&c);
    for (k = 0; k < 2000; ++k) {
        float tau[3], wd[3], dq[4], qn[4], th[3], Iw[3], gy[3];
        alloc_t out;
        att_ctrl_step(&c, &g, qd, q, w, 0.01f, tau);
        alloc_torque(tau, 3.0e6f, off, vb, 1.2f, 0.3f, 0, &out);
        for (i = 0; i < 3; ++i) { Iw[i] = I[i] * w[i]; }
        v3_cross(gy, w, Iw);
        for (i = 0; i < 3; ++i) { wd[i] = (out.tau_ach[i] - gy[i]) / I[i]; w[i] += wd[i] * 0.01f; }
        v3_scale(th, w, 0.01f);
        q_from_rotvec(dq, th); q_mul(qn, q, dq); q_normalize(qn); q_copy(q, qn);
        t += 0.01f;
        if (t > 8.0f) {
            float e[3]; q_error(e, q, qd);
            if (v3_norm(e) > max_over) { max_over = v3_norm(e); }
        }
    }
    { float e[3]; q_error(e, q, qd); th_final = v3_norm(e); }
    printf("    flip: residual %.3f deg, max error after 8 s %.2f deg\n",
           th_final * GNC_RAD2DEG, max_over * GNC_RAD2DEG);
    CHECK(th_final < 0.2f * GNC_DEG2RAD);
    CHECK(max_over < 3.0f * GNC_DEG2RAD);
}

void run_gnc_tests(void);
void run_gnc_tests(void)
{
    RUN(test_pdg_tgo_is_global_minimum);
    RUN(test_pdg_closed_loop_point_mass);
    RUN(test_thrust_projection);
    RUN(test_glide_axis_inverts_crossflow);
    RUN(test_ignition_altitude_engine_aware);
    RUN(test_stopping_margin);
    RUN(test_terminal_descent_profile);
    RUN(test_gimbal_allocation_signs);
    RUN(test_engine_out_offset_cancelled);
    RUN(test_flap_allocation_trim_and_bounds);
    RUN(test_attitude_slew_closed_loop);
}
