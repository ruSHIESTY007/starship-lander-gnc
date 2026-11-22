/** @file test_math.c  @brief Unit tests for gnc_math. */
#include "gnc_math.h"
#include "test_framework.h"

static void test_vec_basics(void)
{
    float a[3] = {1, 2, 3}, b[3] = {-2, 0.5f, 4}, c[3];
    v3_cross(c, a, b);
    CHECK_NEAR(c[0], 2 * 4 - 3 * 0.5, 1e-6);
    CHECK_NEAR(c[1], 3 * -2 - 1 * 4, 1e-6);
    CHECK_NEAR(c[2], 1 * 0.5 - 2 * -2, 1e-6);
    CHECK_NEAR(v3_dot(c, a), 0.0, 1e-5);
    CHECK_NEAR(v3_normalize(c, a), sqrt(14.0), 1e-5);
    CHECK_NEAR(v3_norm(c), 1.0, 1e-6);
    { float z[3] = {0, 0, 0}; CHECK_NEAR(v3_normalize(c, z), 0.0, 0); CHECK(c[0] == 0.0f); }
}

static void test_quat_rotation_matches_dcm(void)
{
    float q[4], R[9], v[3] = {0.3f, -1.2f, 2.0f}, r1[3], r2[3], back[3];
    q_from_euler321(q, 0.7f, -0.4f, 1.1f);
    q_to_dcm(R, q);
    q_rotate(r1, q, v);
    m3_mul_v(r2, R, v);
    CHECK_NEAR(r1[0], r2[0], 1e-5); CHECK_NEAR(r1[1], r2[1], 1e-5); CHECK_NEAR(r1[2], r2[2], 1e-5);
    q_rotate_inv(back, q, r1);
    CHECK_NEAR(back[0], v[0], 1e-5); CHECK_NEAR(back[1], v[1], 1e-5); CHECK_NEAR(back[2], v[2], 1e-5);
}

static void test_euler_roundtrip(void)
{
    float q[4], ypr[3];
    q_from_euler321(q, -2.5f, 0.9f, 0.3f);
    q_to_euler321(ypr, q);
    CHECK_NEAR(ypr[0], -2.5, 1e-5); CHECK_NEAR(ypr[1], 0.9, 1e-5); CHECK_NEAR(ypr[2], 0.3, 1e-5);
}

static void test_dcm_quat_roundtrip(void)
{
    /* sweep orientations hitting all four Shepperd branches */
    int i;
    for (i = 0; i < 64; ++i) {
        float q[4], R[9], q2[4], th[3];
        q_from_euler321(q, 0.37f * (float)i, 0.11f * (float)i - 1.2f, -0.53f * (float)i);
        q_to_dcm(R, q);
        q_from_dcm(q2, R);
        q_error(th, q, q2);
        CHECK_NEAR(v3_norm(th), 0.0, 2e-4);
    }
}

static void test_rotvec_exp_log(void)
{
    float th[3] = {0.4f, -1.3f, 0.8f}, q[4], th2[3], tiny[3] = {1e-8f, -2e-8f, 0}, qt[4];
    q_from_rotvec(q, th);
    q_to_rotvec(th2, q);
    CHECK_NEAR(th2[0], th[0], 1e-5); CHECK_NEAR(th2[1], th[1], 1e-5); CHECK_NEAR(th2[2], th[2], 1e-5);
    q_from_rotvec(qt, tiny);
    CHECK_NEAR(qt[0], 1.0, 1e-7);
}

static void test_q_error_small_angle(void)
{
    float qa[4], dq[4], qb[4], th[3], d[3] = {0.01f, -0.02f, 0.005f};
    q_from_euler321(qa, 1.0f, 0.2f, -0.3f);
    q_from_rotvec(dq, d);
    q_mul(qb, qa, dq);
    q_error(th, qa, qb);
    CHECK_NEAR(th[0], d[0], 1e-5); CHECK_NEAR(th[1], d[1], 1e-5); CHECK_NEAR(th[2], d[2], 1e-5);
}

static void test_att_from_two_axes(void)
{
    /* thrust axis (+x body) straight up, belly (+z body) toward +x_L heading */
    float q[4], up[3] = {0, 0, 1}, hdg[3] = {1, 0.2f, 0}, xb[3], zb[3];
    const float ex[3] = {1, 0, 0}, ez[3] = {0, 0, 1};
    CHECK(att_from_two_axes(q, 0, 1.0f, up, 2, hdg) == 0);
    q_rotate(xb, q, ex);
    q_rotate(zb, q, ez);
    CHECK_NEAR(xb[2], 1.0, 1e-5);
    CHECK_NEAR(zb[2], 0.0, 1e-5);
    CHECK(zb[0] > 0.97f);
    /* degenerate secondary */
    CHECK(att_from_two_axes(q, 0, 1.0f, up, 2, up) == -1);
    /* non-cyclic axis pair (z primary, x secondary) must stay right-handed */
    {
        float down[3] = {0, 0, -1}, R[9], c[3];
        CHECK(att_from_two_axes(q, 2, 1.0f, down, 0, hdg) == 0);
        q_to_dcm(R, q);
        { float x[3] = {R[0], R[3], R[6]}, y[3] = {R[1], R[4], R[7]}, z[3] = {R[2], R[5], R[8]};
          v3_cross(c, x, y);
          CHECK_NEAR(v3_dot(c, z), 1.0, 1e-5);
          CHECK_NEAR(z[2], -1.0, 1e-5); }
    }
}

static void test_chol_solve(void)
{
    float S[9] = {4, 1, 0.5f, 1, 3, 0.2f, 0.5f, 0.2f, 2}, b[3] = {1, -2, 3}, x[3], r[3];
    float NPD[4] = {1, 2, 2, 1};
    CHECK(chol_solve(x, S, b, 3) == 0);
    m3_mul_v(r, S, x);
    CHECK_NEAR(r[0], b[0], 1e-5); CHECK_NEAR(r[1], b[1], 1e-5); CHECK_NEAR(r[2], b[2], 1e-5);
    CHECK(chol_solve(x, NPD, b, 2) == -1);
}

static void test_mat_mul(void)
{
    float A[6] = {1, 2, 3, 4, 5, 6}, B[6] = {1, 0, 0, 1, 1, 1}, C[4], D[4];
    mat_mul(C, A, B, 2, 3, 2);      /* (2x3)(3x2) */
    CHECK_NEAR(C[0], 4, 1e-6); CHECK_NEAR(C[1], 5, 1e-6); CHECK_NEAR(C[2], 10, 1e-6); CHECK_NEAR(C[3], 11, 1e-6);
    mat_mul_abt(D, A, A, 2, 3, 2);  /* A A^T */
    CHECK_NEAR(D[0], 14, 1e-6); CHECK_NEAR(D[1], 32, 1e-6); CHECK_NEAR(D[3], 77, 1e-6);
}

void run_math_tests(void);
void run_math_tests(void)
{
    RUN(test_vec_basics);
    RUN(test_quat_rotation_matches_dcm);
    RUN(test_euler_roundtrip);
    RUN(test_dcm_quat_roundtrip);
    RUN(test_rotvec_exp_log);
    RUN(test_q_error_small_angle);
    RUN(test_att_from_two_axes);
    RUN(test_chol_solve);
    RUN(test_mat_mul);
}
