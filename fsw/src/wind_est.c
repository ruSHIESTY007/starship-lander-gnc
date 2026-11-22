/** @file wind_est.c  @brief Wind / drag-scale EKF. See wind_est.h. */
#include "wind_est.h"

#include <math.h>

#include "aero_model.h"
#include "fsw_config.h"
#include "gnc_math.h"

void wind_init(wind_est_t *we, float sigma_w, float sigma_k)
{
    int i;
    for (i = 0; i < 9; ++i) { we->P[i] = 0.0f; }
    we->x[0] = 0.0f; we->x[1] = 0.0f; we->x[2] = 1.0f;
    we->P[0] = sigma_w * sigma_w;
    we->P[4] = sigma_w * sigma_w;
    we->P[8] = sigma_k * sigma_k;
    we->h_ref = 0.0f;
    we->last_nis = 0.0f;
    we->n_update = 0;
    we->n_reject = 0;
    we->active = 0;
}

void wind_predict(wind_est_t *we, float dt)
{
    we->P[0] += WIND_Q_W * WIND_Q_W * dt;
    we->P[4] += WIND_Q_W * WIND_Q_W * dt;
    we->P[8] += WIND_Q_K * WIND_Q_K * dt;
}

static void predict_meas(const wind_est_t *we, const float x[3], const nav_t *nav,
                         const float f_thr[3], float mass, const float flaps[4],
                         float rho, float a, float out[3], float *qbar, float F_aero[3])
{
    float vrel[3], vb[3];
    int i;
    (void)we;
    vrel[0] = nav->v[0] - x[0];
    vrel[1] = nav->v[1] - x[1];
    vrel[2] = nav->v[2];
    q_rotate_inv(vb, nav->q, vrel);
    *qbar = aero_forces(vb, nav->omega_b, rho, a, flaps, x[2], F_aero, 0);
    for (i = 0; i < 3; ++i) { out[i] = (F_aero[i] + f_thr[i]) / mass; }
}

int wind_update(wind_est_t *we, const nav_t *nav, const float f_thrust_b[3], float mass,
                const float flaps[4])
{
    float rho, a, h0[3], hp[3], H[9], S[9], PHt[9], Fa[3], Fa_p[3], r[3], y[3], rv[3];
    float qbar, qb_p, nis, thr;
    int i, j, k;
    const float step[3] = { 0.5f, 0.5f, 0.02f };

    we->active = 0;
    if (mass < 1.0f) { return 0; }
    atmo_us76(nav->p[2], &rho, &a);
    predict_meas(we, we->x, nav, f_thrust_b, mass, flaps, rho, a, h0, &qbar, Fa);
    if (qbar < WIND_QBAR_MIN) { return 0; }

    for (j = 0; j < 3; ++j) {
        float xp[3];
        xp[0] = we->x[0]; xp[1] = we->x[1]; xp[2] = we->x[2];
        xp[j] += step[j];
        predict_meas(we, xp, nav, f_thrust_b, mass, flaps, rho, a, hp, &qb_p, Fa_p);
        for (i = 0; i < 3; ++i) { H[i * 3 + j] = (hp[i] - h0[i]) / step[j]; }
    }
    /* measurement noise: aero model error + thrust reconstruction + IMU */
    thr = v3_norm(f_thrust_b) / mass;
    for (i = 0; i < 3; ++i) {
        const float ea = WIND_AERO_MODEL_ERR * fabsf(Fa[i]) / mass + 0.02f * v3_norm(Fa) / mass;
        const float et = ((i == 0) ? WIND_THRUST_ERR_AX : WIND_THRUST_ERR_LAT) * thr;
        rv[i] = ea * ea + et * et + 0.01f * 0.01f;
        r[i] = nav->f_b[i] - h0[i];
    }
    /* S = H P H^T + R */
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            float acc = 0.0f;
            for (k = 0; k < 3; ++k) { acc += we->P[i * 3 + k] * H[j * 3 + k]; }
            PHt[i * 3 + j] = acc;
        }
    }
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            float acc = 0.0f;
            for (k = 0; k < 3; ++k) { acc += H[i * 3 + k] * PHt[k * 3 + j]; }
            S[i * 3 + j] = acc + ((i == j) ? rv[i] : 0.0f);
        }
    }
    if (chol_solve(y, S, r, 3) != 0) { return 0; }
    nis = r[0] * y[0] + r[1] * y[1] + r[2] * y[2];
    we->last_nis = nis;
    if (!(nis <= 4.0f * NAV_CHI2_3DOF)) {   /* loose gate: reject gross transients only */
        we->n_reject++;
        return 0;
    }
    /* K = P H^T S^-1 ; solve column-wise */
    {
        float K[9], col[3], e[3];
        for (j = 0; j < 3; ++j) {
            /* row j of K = (S^-1 (P H^T)^T row j)^T  -> solve S k = PHt[j,:]^T */
            e[0] = PHt[j * 3 + 0]; e[1] = PHt[j * 3 + 1]; e[2] = PHt[j * 3 + 2];
            if (chol_solve(col, S, e, 3) != 0) { return 0; }
            K[j * 3 + 0] = col[0]; K[j * 3 + 1] = col[1]; K[j * 3 + 2] = col[2];
        }
        for (i = 0; i < 3; ++i) {
            we->x[i] += K[i * 3 + 0] * r[0] + K[i * 3 + 1] * r[1] + K[i * 3 + 2] * r[2];
        }
        /* Joseph form P = (I-KH) P (I-KH)^T + K R K^T */
        {
            float IKH[9], T1[9], Pn[9];
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    float acc = (i == j) ? 1.0f : 0.0f;
                    for (k = 0; k < 3; ++k) { acc -= K[i * 3 + k] * H[k * 3 + j]; }
                    IKH[i * 3 + j] = acc;
                }
            }
            m3_mul(T1, IKH, we->P);
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    float acc = 0.0f;
                    for (k = 0; k < 3; ++k) {
                        acc += T1[i * 3 + k] * IKH[j * 3 + k] + K[i * 3 + k] * rv[k] * K[j * 3 + k];
                    }
                    Pn[i * 3 + j] = acc;
                }
            }
            mat_symmetrize(Pn, 3);
            for (i = 0; i < 9; ++i) { we->P[i] = Pn[i]; }
        }
    }
    we->x[2] = gnc_clampf(we->x[2], 0.5f, 1.5f);
    we->x[0] = gnc_clampf(we->x[0], -60.0f, 60.0f);
    we->x[1] = gnc_clampf(we->x[1], -60.0f, 60.0f);
    we->h_ref = nav->p[2];
    we->n_update++;
    we->active = 1;
    return 1;
}

static float profile_shape(float h)
{
    if (h >= 1000.0f) { return powf(100.0f, 1.0f / 7.0f); }
    if (h < 0.5f) { h = 0.5f; }
    return powf(h / 10.0f, 1.0f / 7.0f);
}

void wind_at_altitude(const wind_est_t *we, float h, float w_out[3])
{
    const float s = profile_shape(h) / profile_shape(we->h_ref);
    w_out[0] = we->x[0] * s;
    w_out[1] = we->x[1] * s;
    w_out[2] = 0.0f;
}
