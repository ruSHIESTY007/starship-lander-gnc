/**
 * @file nav.c
 * @brief Error-state Kalman filter implementation. See nav.h for the formulation.
 */
#include "nav.h"

#include <math.h>

#include "fsw_config.h"
#include "gnc_math.h"

#define NX NAV_NX

/* scratch buffers (single nav instance per executive; not re-entrant by design) */
static float s_Phi[NX * NX];
static float s_T[NX * NX];

static void set_block3(float *A, int r, int c, const float B[9], float s)
{
    int i, j;
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            A[(r + i) * NX + (c + j)] = s * B[3 * i + j];
        }
    }
}

void nav_init(nav_t *nav, const float p0[3], const float v0[3], const float q0[4],
              const nav_init_sigma_t *sig)
{
    int i;
    const float var[5] = { sig->pos * sig->pos, sig->vel * sig->vel, sig->att * sig->att,
                           sig->gyro_bias * sig->gyro_bias, sig->accel_bias * sig->accel_bias };
    v3_copy(nav->p, p0);
    v3_copy(nav->v, v0);
    q_copy(nav->q, q0);
    (void)q_normalize(nav->q);
    v3_zero(nav->bg);
    v3_zero(nav->ba);
    v3_zero(nav->omega_b);
    v3_zero(nav->f_b);
    mat_zero(nav->P, NX, NX);
    for (i = 0; i < NX; ++i) {
        nav->P[i * NX + i] = var[i / 3];
        nav->dx[i] = 0.0f;
    }
    for (i = 0; i < NAV_N_SENS; ++i) {
        nav->health[i].consec_reject = 0;
        nav->health[i].failed = 0;
        nav->health[i].n_accept = 0;
        nav->health[i].n_reject = 0;
        nav->health[i].last_nis = 0.0f;
    }
    nav->t = 0.0f;
}

void nav_propagate(nav_t *nav, const float gyro[3], const float accel[3], float dt)
{
    float R[9], S[9], RS[9], aL[3], dq[4], th[3], qn[4];
    int i;
    const float g[3] = { 0.0f, 0.0f, -FSW_G0 };

    v3_sub(nav->omega_b, gyro, nav->bg);
    v3_sub(nav->f_b, accel, nav->ba);
    q_to_dcm(R, nav->q);

    /* ---- nominal state (trapezoid-free first-order strapdown, 100 Hz) ---- */
    m3_mul_v(aL, R, nav->f_b);
    v3_add(aL, aL, g);
    v3_axpy(nav->p, dt, nav->v);
    v3_axpy(nav->p, 0.5f * dt * dt, aL);
    v3_axpy(nav->v, dt, aL);
    v3_scale(th, nav->omega_b, dt);
    q_from_rotvec(dq, th);
    q_mul(qn, nav->q, dq);
    (void)q_normalize(qn);
    q_copy(nav->q, qn);

    /* ---- error-state transition Phi = I + F dt ---- */
    mat_identity(s_Phi, NX);
    for (i = 0; i < 3; ++i) {
        s_Phi[(NAV_IP + i) * NX + (NAV_IV + i)] = dt;            /* dp/dv          */
        s_Phi[(NAV_ITH + i) * NX + (NAV_IBG + i)] = -dt;         /* dth/dbg        */
    }
    v3_skew(S, nav->f_b);
    m3_mul(RS, R, S);
    set_block3(s_Phi, NAV_IV, NAV_ITH, RS, -dt);                 /* dv/dth = -R[f]x */
    set_block3(s_Phi, NAV_IV, NAV_IBA, R, -dt);                  /* dv/dba = -R     */
    v3_skew(S, nav->omega_b);
    for (i = 0; i < 9; ++i) { S[i] = -dt * S[i]; }
    S[0] += 1.0f; S[4] += 1.0f; S[8] += 1.0f;
    set_block3(s_Phi, NAV_ITH, NAV_ITH, S, 1.0f);                /* I - [w]x dt     */

    /* ---- P = Phi P Phi^T + Qd ---- */
    mat_mul(s_T, s_Phi, nav->P, NX, NX, NX);
    mat_mul_abt(nav->P, s_T, s_Phi, NX, NX, NX);
    {
        const float qv = (NAV_ACCEL_VRW * NAV_ACCEL_VRW + NAV_VEL_PROC_EXTRA * NAV_VEL_PROC_EXTRA) * dt;
        const float qth = NAV_GYRO_ARW * NAV_GYRO_ARW * dt;
        const float qbg = NAV_GYRO_BIAS_RW * NAV_GYRO_BIAS_RW * dt;
        const float qba = NAV_ACCEL_BIAS_RW * NAV_ACCEL_BIAS_RW * dt;
        const float qp = NAV_POS_PROC * NAV_POS_PROC * dt;
        for (i = 0; i < 3; ++i) {
            nav->P[(NAV_IP + i) * (NX + 1)] += qp;
            nav->P[(NAV_IV + i) * (NX + 1)] += qv;
            nav->P[(NAV_ITH + i) * (NX + 1)] += qth;
            nav->P[(NAV_IBG + i) * (NX + 1)] += qbg;
            nav->P[(NAV_IBA + i) * (NX + 1)] += qba;
        }
    }
    mat_symmetrize(nav->P, NX);
    nav->t += dt;
}

/** Inject the accumulated error state into the nominal state and reset it. */
static void inject(nav_t *nav)
{
    float dq[4], qn[4];
    int i;
    for (i = 0; i < 3; ++i) {
        nav->p[i] += nav->dx[NAV_IP + i];
        nav->v[i] += nav->dx[NAV_IV + i];
        nav->bg[i] += nav->dx[NAV_IBG + i];
        nav->ba[i] += nav->dx[NAV_IBA + i];
    }
    q_from_rotvec(dq, &nav->dx[NAV_ITH]);
    q_mul(qn, nav->q, dq);
    (void)q_normalize(qn);
    q_copy(nav->q, qn);
    for (i = 0; i < NX; ++i) { nav->dx[i] = 0.0f; }
}

/**
 * Gated vector update with diagonal R processed as sequential scalar updates.
 * H is m x NX (row-major), resid = z - h(x_hat).
 */
static int vector_update(nav_t *nav, int sens, int m, const float *H, const float *resid,
                         const float *rvar, float gate)
{
    static float PHt[NX * 3];
    static float Ph[NX];
    float S[9], y[3];
    float nis;
    nav_health_t *hl = &nav->health[sens];
    int i, j, k;

    if (hl->failed) { return NAV_UPD_SENSOR_FAILED; }

    /* innovation covariance S = H P H^T + R */
    for (i = 0; i < NX; ++i) {
        for (j = 0; j < m; ++j) {
            float acc = 0.0f;
            for (k = 0; k < NX; ++k) { acc += nav->P[i * NX + k] * H[j * NX + k]; }
            PHt[i * m + j] = acc;
        }
    }
    for (i = 0; i < m; ++i) {
        for (j = 0; j < m; ++j) {
            float acc = 0.0f;
            for (k = 0; k < NX; ++k) { acc += H[i * NX + k] * PHt[k * m + j]; }
            S[i * m + j] = acc + ((i == j) ? rvar[i] : 0.0f);
        }
    }
    if (chol_solve(y, S, resid, m) != 0) { return NAV_UPD_INVALID; }
    nis = 0.0f;
    for (i = 0; i < m; ++i) { nis += resid[i] * y[i]; }
    hl->last_nis = nis;

    if (!(nis <= gate)) {          /* also catches NaN */
        hl->n_reject++;
        hl->consec_reject++;
        if (hl->consec_reject >= NAV_FAIL_CONSEC) { hl->failed = 1; }
        return NAV_UPD_REJECTED;
    }
    hl->consec_reject = 0;
    hl->n_accept++;

    /* sequential scalar updates */
    for (j = 0; j < m; ++j) {
        const float *h = &H[j * NX];
        float s = rvar[j];
        float innov = resid[j];
        for (i = 0; i < NX; ++i) {
            float acc = 0.0f;
            for (k = 0; k < NX; ++k) { acc += nav->P[i * NX + k] * h[k]; }
            Ph[i] = acc;
            innov -= h[i] * nav->dx[i];
        }
        for (i = 0; i < NX; ++i) { s += h[i] * Ph[i]; }
        if (s <= 0.0f) { return NAV_UPD_INVALID; }
        for (i = 0; i < NX; ++i) { nav->dx[i] += Ph[i] * innov / s; }
        for (i = 0; i < NX; ++i) {
            for (k = 0; k < NX; ++k) { nav->P[i * NX + k] -= Ph[i] * Ph[k] / s; }
        }
    }
    mat_symmetrize(nav->P, NX);
    for (i = 0; i < NX; ++i) {            /* guard float32 round-off */
        if (nav->P[i * NX + i] < 1.0e-12f) { nav->P[i * NX + i] = 1.0e-12f; }
    }
    inject(nav);
    return NAV_UPD_ACCEPTED;
}

int nav_update_gnss_pos(nav_t *nav, const float pos[3])
{
    float H[3 * NX], r[3], rv[3];
    int i;
    mat_zero(H, 3, NX);
    for (i = 0; i < 3; ++i) {
        H[i * NX + NAV_IP + i] = 1.0f;
        r[i] = pos[i] - nav->p[i];
    }
    rv[0] = NAV_GNSS_POS_SIG_H * NAV_GNSS_POS_SIG_H;
    rv[1] = rv[0];
    rv[2] = NAV_GNSS_POS_SIG_V * NAV_GNSS_POS_SIG_V;
    return vector_update(nav, NAV_SENS_GNSS_POS, 3, H, r, rv, NAV_CHI2_3DOF);
}

int nav_update_gnss_vel(nav_t *nav, const float vel[3])
{
    float H[3 * NX], r[3], rv[3];
    int i;
    mat_zero(H, 3, NX);
    for (i = 0; i < 3; ++i) {
        H[i * NX + NAV_IV + i] = 1.0f;
        r[i] = vel[i] - nav->v[i];
        rv[i] = NAV_GNSS_VEL_SIG * NAV_GNSS_VEL_SIG;
    }
    return vector_update(nav, NAV_SENS_GNSS_VEL, 3, H, r, rv, NAV_CHI2_3DOF);
}

int nav_radar_model(const nav_t *nav, int32_t antenna, float *range_pred, float H[NAV_NX])
{
    /* boresight b_B: belly = +z_B, aft = -x_B.  c = cos(off-nadir) = -e3 . (R b) */
    float R[9], b[3], bL[3], g[3], gxb[3];
    float c, pz;
    int i;
    if (antenna == 0) { v3_set(b, 0.0f, 0.0f, 1.0f); } else { v3_set(b, -1.0f, 0.0f, 0.0f); }
    q_to_dcm(R, nav->q);
    m3_mul_v(bL, R, b);
    c = -bL[2];
    if (c < NAV_RADAR_MAX_OFFNADIR_COS) { return -1; }
    pz = nav->p[2];
    *range_pred = pz / c;
    for (i = 0; i < NAV_NX; ++i) { H[i] = 0.0f; }
    H[NAV_IP + 2] = 1.0f / c;
    /* dc/dth = (g x b)^T with g = R^T e3 ; drange/dth = -pz/c^2 dc/dth */
    v3_set(g, R[6], R[7], R[8]);
    v3_cross(gxb, g, b);
    for (i = 0; i < 3; ++i) { H[NAV_ITH + i] = -pz / (c * c) * gxb[i]; }
    return 0;
}

int nav_update_radar(nav_t *nav, float range, int32_t antenna)
{
    float H[NX], pred, r, rv;
    if (nav_radar_model(nav, antenna, &pred, H) != 0) { return NAV_UPD_INVALID; }
    r = range - pred;
    rv = NAV_RADAR_SIG_ABS + NAV_RADAR_SIG_REL * fabsf(range);
    rv = rv * rv;
    return vector_update(nav, NAV_SENS_RADAR, 1, H, &r, &rv, NAV_CHI2_1DOF);
}
