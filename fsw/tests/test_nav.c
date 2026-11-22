/**
 * @file test_nav.c
 * @brief ESKF verification: Jacobian checks, convergence + covariance consistency (NEES)
 *        on a synthetic manoeuvring trajectory, fault rejection / latching.
 */
#include <math.h>
#include <stdio.h>

#include "fsw_config.h"
#include "gnc_math.h"
#include "nav.h"
#include "test_framework.h"

/* ---------------- deterministic Gaussian RNG ---------------- */
static unsigned long long s_rng = 88172645463325252ULL;
static double urand(void)
{
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return ((double)(s_rng >> 11) + 0.5) / 9007199254740992.0;
}
static double nrand(void)
{
    return sqrt(-2.0 * log(urand())) * cos(6.283185307179586 * urand());
}

/* ---------------- double-precision truth kinematics ---------------- */
typedef struct { double p[3], v[3], q[4]; } truth_t;

static void dq_mul(double o[4], const double a[4], const double b[4])
{
    double t[4];
    t[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    t[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    t[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    t[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    o[0] = t[0]; o[1] = t[1]; o[2] = t[2]; o[3] = t[3];
}
static void dq_rot(double o[3], const double q[4], const double v[3])
{
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double R[9] = { 1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                          2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                          2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y) };
    o[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    o[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    o[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}
static void omega_of(double t, double w[3])
{
    w[0] = 0.10 * sin(0.30 * t); w[1] = 0.06 * cos(0.20 * t); w[2] = 0.08 * sin(0.11 * t + 1.0);
}
static void fL_of(double t, double f[3])
{
    /* specific force in L: hover-like plus manoeuvres (thrust-like excitation) */
    f[0] = 2.0 * sin(0.10 * t); f[1] = 1.5 * cos(0.13 * t); f[2] = 9.80665 + 3.0 * sin(0.05 * t);
}
static void truth_step(truth_t *x, double t, double dt, double fb_out[3], double w_out[3])
{
    /* midpoint integration with 10 sub-steps; returns body rate / specific force at t+dt */
    int s;
    const double h = dt / 10.0;
    for (s = 0; s < 10; ++s) {
        double tm = t + (s + 0.5) * h, w[3], f[3], dq[4], a;
        omega_of(tm, w);
        fL_of(tm, f);
        f[2] -= 9.80665;
        x->p[0] += x->v[0] * h + 0.5 * f[0] * h * h;
        x->p[1] += x->v[1] * h + 0.5 * f[1] * h * h;
        x->p[2] += x->v[2] * h + 0.5 * f[2] * h * h;
        x->v[0] += f[0] * h; x->v[1] += f[1] * h; x->v[2] += f[2] * h;
        a = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]) * h;
        dq[0] = cos(0.5 * a);
        if (a > 1e-12) {
            double s2 = sin(0.5 * a) / (a / h);
            dq[1] = s2 * w[0]; dq[2] = s2 * w[1]; dq[3] = s2 * w[2];
        } else { dq[1] = dq[2] = dq[3] = 0.0; }
        dq_mul(x->q, x->q, dq);
    }
    {
        double fL[3], qc[4];
        omega_of(t + dt, w_out);
        fL_of(t + dt, fL);
        qc[0] = x->q[0]; qc[1] = -x->q[1]; qc[2] = -x->q[2]; qc[3] = -x->q[3];
        dq_rot(fb_out, qc, fL);
    }
}

/* ---------------- tests ---------------- */
static void test_radar_jacobian_numeric(void)
{
    nav_t nav;
    nav_init_sigma_t sig = { 1, 1, 0.01f, 1e-5f, 1e-2f };
    float p0[3] = { 10, 20, 800 }, v0[3] = { 0, 0, -50 }, q0[4];
    float H[NAV_NX], r0, r1, dth[3];
    int ant, j;
    for (ant = 0; ant < 2; ++ant) {
        if (ant == 0) { q_from_euler321(q0, 0.2f, 0.1f, GNC_PI + 0.15f); }
        else { q_from_euler321(q0, 0.3f, -GNC_PI / 2 + 0.25f, 0.1f); }
        nav_init(&nav, p0, v0, q0, &sig);
        CHECK(nav_radar_model(&nav, ant, &r0, H) == 0);
        for (j = 0; j < 3; ++j) {
            /* central difference: float32 range ~800 m needs a large step */
            nav_t n2 = nav, n3 = nav;
            float dq[4], qn[4], Hd[NAV_NX], rm;
            const float hstep = 1e-2f;
            v3_zero(dth); dth[j] = hstep;
            q_from_rotvec(dq, dth); q_mul(qn, nav.q, dq); q_copy(n2.q, qn);
            (void)nav_radar_model(&n2, ant, &r1, Hd);
            dth[j] = -hstep;
            q_from_rotvec(dq, dth); q_mul(qn, nav.q, dq); q_copy(n3.q, qn);
            (void)nav_radar_model(&n3, ant, &rm, Hd);
            CHECK_NEAR((r1 - rm) / (2.0f * hstep), H[NAV_ITH + j], 0.01 * fabs(H[NAV_ITH + j]) + 0.02);
        }
        {
            nav_t n2 = nav;
            n2.p[2] += 1.0f;
            (void)nav_radar_model(&n2, ant, &r1, H);
            CHECK_NEAR(r1 - r0, 1.0 / cos(0.0) * (r0 / 800.0), 1e-3);
        }
    }
    /* boresight too far from nadir must be refused */
    q_identity(q0);   /* belly up, nose horizontal: neither antenna sees ground */
    nav_init(&nav, p0, v0, q0, &sig);
    CHECK(nav_radar_model(&nav, 0, &r0, H) == -1);
    CHECK(nav_radar_model(&nav, 1, &r0, H) == -1);
}

typedef struct { double nees_p, nees_v, nees_th; int n; double max_pos_err, max_att_err; } nees_t;

static void run_filter(int gnss_fault, nees_t *out, nav_t *nav_out)
{
    truth_t x = { { -500, 200, 3000 }, { 40, -5, -60 }, { 1, 0, 0, 0 } };
    const double bg[3] = { 3e-5, -2e-5, 4e-5 }, ba[3] = { 0.012, -0.008, 0.01 };
    nav_t nav;
    nav_init_sigma_t sig = { 5.0f, 0.5f, 0.02f, 5e-5f, 0.02f };
    const float dt = 0.01f;
    float p0[3], v0[3], q0[4], dth0[3];
    double t = 0.0;
    int k, i;
    const int N = 12000;   /* 120 s */
    nees_t acc = { 0, 0, 0, 0, 0, 0 };

    s_rng = 88172645463325252ULL;
    for (i = 0; i < 3; ++i) {
        p0[i] = (float)(x.p[i] + 5.0 * nrand());
        v0[i] = (float)(x.v[i] + 0.5 * nrand());
        dth0[i] = (float)(0.02 * nrand());
    }
    {   float qt[4] = { (float)x.q[0], (float)x.q[1], (float)x.q[2], (float)x.q[3] }, dq[4];
        q_from_rotvec(dq, dth0); q_mul(q0, qt, dq); }
    nav_init(&nav, p0, v0, q0, &sig);

    for (k = 0; k < N; ++k) {
        double fb[3], w[3];
        float gyro[3], acc_m[3];
        truth_step(&x, t, dt, fb, w);
        t += dt;
        for (i = 0; i < 3; ++i) {
            gyro[i] = (float)(w[i] + bg[i] + NAV_GYRO_ARW / sqrt(dt) * nrand());
            acc_m[i] = (float)(fb[i] + ba[i] + NAV_ACCEL_VRW / sqrt(dt) * nrand());
        }
        nav_propagate(&nav, gyro, acc_m, dt);
        if (k % 10 == 0) {
            float pm[3], vm[3];
            for (i = 0; i < 3; ++i) {
                const double sp = (i < 2) ? 1.0 : 2.0;
                pm[i] = (float)(x.p[i] + sp * nrand());
                vm[i] = (float)(x.v[i] + 0.05 * nrand());
            }
            if (gnss_fault && t > 60.0) { pm[0] += 40.0f; }
            (void)nav_update_gnss_pos(&nav, pm);
            (void)nav_update_gnss_vel(&nav, vm);
        }
        if (k > 3000) {   /* after convergence: accumulate NEES per 3-dof block */
            double e[9];
            float qt[4] = { (float)x.q[0], (float)x.q[1], (float)x.q[2], (float)x.q[3] }, th[3];
            q_error(th, nav.q, qt);
            for (i = 0; i < 3; ++i) {
                e[i] = x.p[i] - nav.p[i];
                e[3 + i] = x.v[i] - nav.v[i];
                e[6 + i] = th[i];
            }
            {
                int b;
                for (b = 0; b < 3; ++b) {
                    float S[9], r[3], y[3];
                    int r_, c_;
                    for (r_ = 0; r_ < 3; ++r_) {
                        r[r_] = (float)e[3 * b + r_];
                        for (c_ = 0; c_ < 3; ++c_) { S[3 * r_ + c_] = nav.P[(3 * b + r_) * NAV_NX + 3 * b + c_]; }
                    }
                    if (chol_solve(y, S, r, 3) == 0) {
                        const double nees = r[0] * y[0] + r[1] * y[1] + r[2] * y[2];
                        if (b == 0) acc.nees_p += nees; else if (b == 1) acc.nees_v += nees; else acc.nees_th += nees;
                    }
                }
            }
            {
                double pe = sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
                double ae = sqrt(e[6]*e[6] + e[7]*e[7] + e[8]*e[8]);
                if (pe > acc.max_pos_err) acc.max_pos_err = pe;
                if (ae > acc.max_att_err) acc.max_att_err = ae;
            }
            acc.n++;
        }
    }
    acc.nees_p /= acc.n; acc.nees_v /= acc.n; acc.nees_th /= acc.n;
    *out = acc;
    *nav_out = nav;
    (void)bg; (void)ba;
}

static void test_convergence_and_consistency(void)
{
    nees_t r;
    nav_t nav;
    run_filter(0, &r, &nav);
    printf("    mean NEES pos=%.2f vel=%.2f att=%.2f (3 = ideal), max |dp|=%.2f m, max |dth|=%.2f mrad\n",
           r.nees_p, r.nees_v, r.nees_th, r.max_pos_err, 1e3 * r.max_att_err);
    printf("    bias est: bg=[%.1e %.1e %.1e] ba=[%.4f %.4f %.4f]\n",
           nav.bg[0], nav.bg[1], nav.bg[2], nav.ba[0], nav.ba[1], nav.ba[2]);
    /* GNSS correlated bias is absent here and R is inflated for it -> NEES below 3 expected */
    CHECK(r.nees_p > 0.3 && r.nees_p < 6.0);
    CHECK(r.nees_v > 0.3 && r.nees_v < 6.0);
    CHECK(r.nees_th > 0.3 && r.nees_th < 6.0);
    CHECK(r.max_pos_err < 4.0);
    CHECK(r.max_att_err < 5e-3);   /* yaw weakly observable under ~2 m/s^2 lateral accel */
    /* accel bias is observable under the manoeuvres */
    CHECK_NEAR(nav.ba[0], 0.012, 0.004);
    CHECK_NEAR(nav.ba[1], -0.008, 0.004);
    CHECK_NEAR(nav.ba[2], 0.010, 0.004);
    CHECK(nav.health[NAV_SENS_GNSS_POS].failed == 0);
}

static void test_gnss_step_fault_is_latched(void)
{
    nees_t r;
    nav_t nav;
    run_filter(1, &r, &nav);
    CHECK(nav.health[NAV_SENS_GNSS_POS].failed == 1);
    CHECK(nav.health[NAV_SENS_GNSS_VEL].failed == 0);
    /* the 40 m step must not have been absorbed into the solution */
    printf("    after GNSS step fault: max |dp| = %.2f m, rejects = %d\n", r.max_pos_err,
           (int)nav.health[NAV_SENS_GNSS_POS].n_reject);
    CHECK(r.max_pos_err < 8.0);
}

static void test_radar_update_corrects_altitude(void)
{
    nav_t nav;
    nav_init_sigma_t sig = { 20.0f, 1.0f, 0.01f, 1e-5f, 1e-2f };
    float p0[3] = { 0, 0, 500 }, v0[3] = { 0, 0, -10 }, q0[4];
    int k, res = 0;
    q_from_euler321(q0, 0.0f, -GNC_PI / 2, 0.0f);   /* nose up, aft antenna to nadir */
    nav_init(&nav, p0, v0, q0, &sig);
    for (k = 0; k < 20; ++k) { res = nav_update_radar(&nav, 520.0f, 1); }
    CHECK(res == NAV_UPD_ACCEPTED);
    CHECK_NEAR(nav.p[2], 520.0, 1.0);
    /* a stuck/garbage radar far outside the covariance is rejected then latched */
    for (k = 0; k < NAV_FAIL_CONSEC; ++k) { res = nav_update_radar(&nav, 50.0f, 1); }
    CHECK(res == NAV_UPD_REJECTED);
    CHECK(nav.health[NAV_SENS_RADAR].failed == 1);
    CHECK(nav_update_radar(&nav, 520.0f, 1) == NAV_UPD_SENSOR_FAILED);
}

void run_nav_tests(void);
void run_nav_tests(void)
{
    RUN(test_radar_jacobian_numeric);
    RUN(test_convergence_and_consistency);
    RUN(test_gnss_step_fault_is_latched);
    RUN(test_radar_update_corrects_altitude);
}
