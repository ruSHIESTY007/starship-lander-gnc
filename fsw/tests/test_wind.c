/**
 * @file test_wind.c
 * @brief Wind estimator: convergence on a synthetic belly-flop glide and on a powered,
 *        tail-down descent (lateral wind vs. lateral thrust), plus the profile extrapolation.
 */
#include <math.h>
#include <stdio.h>

#include "aero_model.h"
#include "fsw_config.h"
#include "gnc_math.h"
#include "nav.h"
#include "test_framework.h"
#include "wind_est.h"

static unsigned long long s_r = 1234567ULL;
static float gauss(void)
{
    double u1, u2;
    s_r ^= s_r << 13; s_r ^= s_r >> 7; s_r ^= s_r << 17; u1 = ((double)(s_r >> 11) + 0.5) / 9007199254740992.0;
    s_r ^= s_r << 13; s_r ^= s_r >> 7; s_r ^= s_r << 17; u2 = ((double)(s_r >> 11) + 0.5) / 9007199254740992.0;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2));
}

static void run_case(const float q[4], const float vL[3], const float thrust_b[3], float mass,
                     const float w_true[3], float k_true, int n, float w_est[2], float *k_est)
{
    nav_t nav;
    wind_est_t we;
    nav_init_sigma_t sig = { 1, 0.1f, 0.01f, 1e-5f, 1e-3f };
    const float p0[3] = { 0, 0, 3000 };
    const float flaps[4] = { 0.2f, 0.2f, 0.73f, 0.73f };
    int k;
    nav_init(&nav, p0, vL, q, &sig);
    wind_init(&we, 15.0f, 0.2f);
    for (k = 0; k < n; ++k) {
        float rho, a, vrel[3], vb[3], F[3];
        int i;
        atmo_us76(nav.p[2], &rho, &a);
        v3_sub(vrel, vL, w_true);
        q_rotate_inv(vb, q, vrel);
        (void)aero_forces(vb, nav.omega_b, rho, a, flaps, k_true, F, 0);
        for (i = 0; i < 3; ++i) {
            /* truth specific force with 5 % aero-model mismatch + IMU noise */
            nav.f_b[i] = (1.05f * F[i] + thrust_b[i]) / mass + 0.005f * gauss();
        }
        wind_predict(&we, FSW_DT);
        if (k % 5 == 0) { (void)wind_update(&we, &nav, thrust_b, mass, flaps); }
    }
    w_est[0] = we.x[0]; w_est[1] = we.x[1];
    *k_est = we.x[2];
    printf("    wind est = (%.2f, %.2f) true (%.2f, %.2f); k = %.3f; sigma = (%.2f, %.2f); updates %d\n",
           we.x[0], we.x[1], w_true[0], w_true[1], we.x[2], sqrtf(we.P[0]), sqrtf(we.P[4]), (int)we.n_update);
}

static void test_glide_wind_convergence(void)
{
    /* belly-flop pitched 15 deg nose-up so both horizontal axes are observable */
    float q[4], w_est[2], k_est;
    const float vL[3] = { 20.0f, 5.0f, -75.0f }, thr[3] = { 0, 0, 0 }, w[3] = { 9.0f, -6.0f, 0.0f };
    q_from_euler321(q, 0.3f, 0.26f, GNC_PI);
    run_case(q, vL, thr, VEH_M_WET, w, 1.1f, 3000, w_est, &k_est);
    CHECK_NEAR(w_est[0], w[0], 2.0);
    CHECK_NEAR(w_est[1], w[1], 2.0);
    CHECK_NEAR(k_est, 1.1 * 1.05, 0.08);
}

static void test_powered_descent_lateral_wind(void)
{
    /* tail-down, thrusting: horizontal wind is crossflow on the body */
    float q[4], w_est[2], k_est;
    const float vL[3] = { 2.0f, -1.0f, -40.0f }, w[3] = { -7.0f, 4.0f, 0.0f };
    const float thr[3] = { 1.2e6f, 0.0f, 0.0f };
    q_from_euler321(q, 0.0f, -GNC_PI / 2, 0.0f);
    run_case(q, vL, thr, 110000.0f, w, 1.0f, 3000, w_est, &k_est);
    CHECK_NEAR(w_est[0], w[0], 2.0);
    CHECK_NEAR(w_est[1], w[1], 2.0);
}

static void test_profile_extrapolation(void)
{
    wind_est_t we;
    float w[3];
    wind_init(&we, 10, 0.1f);
    we.x[0] = 10.0f; we.x[1] = 0.0f; we.h_ref = 1000.0f;
    wind_at_altitude(&we, 10.0f, w);
    CHECK_NEAR(w[0], 10.0 / pow(100.0, 1.0 / 7.0), 1e-3);
    wind_at_altitude(&we, 3000.0f, w);
    CHECK_NEAR(w[0], 10.0, 1e-4);
}

void run_wind_tests(void);
void run_wind_tests(void)
{
    RUN(test_glide_wind_convergence);
    RUN(test_powered_descent_lateral_wind);
    RUN(test_profile_extrapolation);
}
