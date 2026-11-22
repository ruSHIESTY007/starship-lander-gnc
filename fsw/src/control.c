/** @file control.c  @brief Attitude control and torque allocation. See control.h. */
#include "control.h"

#include <math.h>

#include "aero_model.h"
#include "fsw_config.h"
#include "gnc_math.h"

static const float I_DIAG[3] = { VEH_IXX, VEH_IYY, VEH_IZZ };

void att_ctrl_reset(att_ctrl_t *c)
{
    v3_zero(c->integ);
    v3_zero(c->th_err);
    v3_zero(c->w_cmd);
    c->hold_integrator = 0;
}

void att_ctrl_step(att_ctrl_t *c, const att_gains_t *g, const float q_des[4], const float q[4],
                   const float w[3], float dt, float tau_req[3])
{
    float th, wmag, lim, Iw[3], gyro[3];
    int i;
    q_error(c->th_err, q, q_des);                 /* rotation current -> desired, body axes */
    th = v3_norm(c->th_err);
    v3_scale(c->w_cmd, c->th_err, g->k_att);
    wmag = g->k_att * th;
    lim = fminf(g->w_max, sqrtf(2.0f * g->alpha_avail * th));
    if (wmag > lim && wmag > 1.0e-9f) { v3_scale(c->w_cmd, c->w_cmd, lim / wmag); }
    for (i = 0; i < 3; ++i) {
        const float e = c->w_cmd[i] - w[i];
        if (!c->hold_integrator) {
            c->integ[i] = gnc_clampf(c->integ[i] + g->k_i * e * dt, -g->i_lim, g->i_lim);
        }
        Iw[i] = I_DIAG[i] * w[i];
        tau_req[i] = I_DIAG[i] * (g->k_rate * e + c->integ[i]);
    }
    v3_cross(gyro, w, Iw);
    v3_add(tau_req, tau_req, gyro);
}

void engine_offset_torque(const float T_each[3], float tau[3])
{
    int i;
    v3_zero(tau);
    for (i = 0; i < FSW_N_ENG; ++i) {
        const float y = VEH_ENG_RING_R * cosf(VEH_ENG_ANGLE[i]);
        const float z = VEH_ENG_RING_R * sinf(VEH_ENG_ANGLE[i]);
        /* r x (T,0,0) = (0, z T, -y T) */
        tau[1] += z * T_each[i];
        tau[2] -= y * T_each[i];
    }
}

/* Solve the flap allocation  B e = r  about e0 with bounds [0,1] (cascaded generalised inverse). */
static void flap_alloc(const float B[12], const float r[3], float e0, float e[4])
{
    int free_[4] = { 1, 1, 1, 1 };
    int it, i, j, k;
    for (i = 0; i < 4; ++i) { e[i] = e0; }
    for (it = 0; it < 4; ++it) {
        float res[3], BBt[9], y[3], lam, tr = 0.0f;
        int any_sat = 0;
        for (i = 0; i < 3; ++i) {
            float acc = r[i];
            for (j = 0; j < 4; ++j) { acc -= B[i * 4 + j] * e[j]; }
            res[i] = acc;
        }
        for (i = 0; i < 3; ++i) {
            for (k = 0; k < 3; ++k) {
                float acc = 0.0f;
                for (j = 0; j < 4; ++j) { if (free_[j]) { acc += B[i * 4 + j] * B[k * 4 + j]; } }
                BBt[i * 3 + k] = acc;
            }
            tr += BBt[i * 4];
        }
        if (tr < 1.0f) { return; }
        lam = 1.0e-3f * tr;
        for (i = 0; i < 3; ++i) { BBt[i * 4] += lam; }
        if (chol_solve(y, BBt, res, 3) != 0) { return; }
        for (j = 0; j < 4; ++j) {
            if (free_[j]) {
                e[j] += B[0 * 4 + j] * y[0] + B[1 * 4 + j] * y[1] + B[2 * 4 + j] * y[2];
            }
        }
        for (j = 0; j < 4; ++j) {
            if (free_[j] && (e[j] < 0.0f || e[j] > 1.0f)) {
                e[j] = gnc_clampf(e[j], 0.0f, 1.0f);
                free_[j] = 0;
                any_sat = 1;
            }
        }
        if (!any_sat) { return; }
    }
}

void alloc_torque(const float tau_act[3], float T_total, const float tau_eng_off[3],
                  const float v_air_b[3], float rho, float flap_trim, int32_t use_flaps,
                  alloc_t *out)
{
    float res[3], tg[3];
    int i;
    v3_copy(res, tau_act);
    /* ---- 1. gimbal ---- */
    out->gimbal[0] = 0.0f;
    out->gimbal[1] = 0.0f;
    v3_copy(tg, tau_eng_off);
    if (T_total > 0.05f * VEH_T_MAX_EACH) {
        /* tau_y = -x_aft T g_z ; tau_z = x_aft T g_y   (r x F at the aft engine plane) */
        float gy = (tau_act[2] - tau_eng_off[2]) / (VEH_X_AFT * T_total);
        float gz = (tau_act[1] - tau_eng_off[1]) / (-VEH_X_AFT * T_total);
        const float gn = sqrtf(gy * gy + gz * gz);
        if (gn > VEH_GIMBAL_MAX) { gy *= VEH_GIMBAL_MAX / gn; gz *= VEH_GIMBAL_MAX / gn; }
        out->gimbal[0] = gy;
        out->gimbal[1] = gz;
        tg[1] += -VEH_X_AFT * T_total * gz;
        tg[2] += VEH_X_AFT * T_total * gy;
    }
    v3_sub(res, res, tg);
    /* ---- 2. flaps ---- */
    {
        float B[12], Fz[4], tf[3] = { 0.0f, 0.0f, 0.0f };
        aero_flap_effectiveness(v_air_b, rho, B, Fz);
        if (use_flaps) {
            flap_alloc(B, res, flap_trim, out->flaps);
        } else {
            for (i = 0; i < 4; ++i) { out->flaps[i] = flap_trim; }
        }
        for (i = 0; i < 3; ++i) {
            tf[i] = B[i * 4 + 0] * out->flaps[0] + B[i * 4 + 1] * out->flaps[1]
                  + B[i * 4 + 2] * out->flaps[2] + B[i * 4 + 3] * out->flaps[3];
        }
        v3_sub(res, res, tf);
        v3_add(tg, tg, tf);
    }
    /* ---- 3. RCS ---- */
    for (i = 0; i < 3; ++i) {
        out->rcs[i] = gnc_clampf(res[i], -VEH_RCS_MAX, VEH_RCS_MAX);
    }
    v3_add(out->tau_ach, tg, out->rcs);
    {
        float d[3];
        v3_sub(d, tau_act, out->tau_ach);
        out->saturated = (v3_norm(d) > 0.05f * v3_norm(tau_act) + 1.0e4f) ? 1 : 0;
    }
}
