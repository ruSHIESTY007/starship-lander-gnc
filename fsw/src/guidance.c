/** @file guidance.c  @brief Guidance laws. See guidance.h for formulation and references. */
#include "guidance.h"

#include <math.h>

#include "fsw_config.h"
#include "gnc_math.h"

/* ============================ glide ============================ */
void glide_axis_for_force(const float v_air[3], const float n_des[3], const float x_cur[3],
                          float x_des[3])
{
    /* For a slender axisymmetric body the crossflow force is -(v - (v.x)x) (direction), i.e.
     * perpendicular to x and in span(v, x). Requiring it along n gives x in span(v, n) with
     * x perpendicular to n:  x ~ v - (v.n) n. The sign of x is free (nose vs tail): pick the
     * one closest to the current nose to minimise slewing. When v is (anti)parallel to n
     * any axis perpendicular to n works: blend toward the current nose projected.          */
    float xr[3], xc[3], n[3];
    float s, V, w;
    (void)v3_normalize(n, n_des);
    V = v3_norm(v_air);
    v3_copy(xr, v_air);
    v3_axpy(xr, -v3_dot(v_air, n), n);
    s = v3_normalize(xr, xr);
    v3_copy(xc, x_cur);
    v3_axpy(xc, -v3_dot(x_cur, n), n);
    if (v3_normalize(xc, xc) < 1.0e-6f) {
        /* current nose along n: any perpendicular */
        float t[3] = { 1.0f, 0.0f, 0.0f };
        v3_axpy(t, -v3_dot(t, n), n);
        (void)v3_normalize(xc, t);
    }
    if (v3_dot(xr, xc) < 0.0f) { v3_scale(xr, xr, -1.0f); }
    w = (V > 1.0f) ? gnc_clampf(s / (V * 0.0523f), 0.0f, 1.0f) : 0.0f;   /* sin 3 deg */
    v3_scale(x_des, xr, w);
    v3_axpy(x_des, 1.0f - w, xc);
    if (v3_normalize(x_des, x_des) < 1.0e-6f) { v3_copy(x_des, xc); }
}

int glide_guidance(const float r[3], const float v[3], const float wind[3], const float aim[2],
                   float h_ign, const float x_cur[3], float q_des[4], float a_h_cmd[2])
{
    float t, zem[2], zev[2], ah[2], amag, amax, n[3], vair[3], xd[3];
    t = (r[2] - h_ign) / fmaxf(-v[2], 20.0f);
    if (t < GLIDE_TGO_MIN) { t = GLIDE_TGO_MIN; }
    zem[0] = aim[0] - r[0] - v[0] * t;
    zem[1] = aim[1] - r[1] - v[1] * t;
    zev[0] = -v[0];
    zev[1] = -v[1];
    ah[0] = 6.0f * zem[0] / (t * t) - 2.0f * zev[0] / t;
    ah[1] = 6.0f * zem[1] / (t * t) - 2.0f * zev[1] / t;
    amag = sqrtf(ah[0] * ah[0] + ah[1] * ah[1]);
    amax = FSW_G0 * tanf(GLIDE_TILT_MAX);
    if (amag > amax) { ah[0] *= amax / amag; ah[1] *= amax / amag; }
    a_h_cmd[0] = ah[0];
    a_h_cmd[1] = ah[1];
    v3_set(n, ah[0], ah[1], FSW_G0);
    (void)v3_normalize(n, n);
    v3_sub(vair, v, wind);
    glide_axis_for_force(vair, n, x_cur, xd);
    /* belly (+z_B) faces into the relative wind */
    return att_from_two_axes(q_des, 0, 1.0f, xd, 2, vair);
}

/* ============================ ignition ============================ */
float ignition_altitude(float vz, float mass, int32_t n_burn, float h_gate)
{
    const float V = fmaxf(-vz, 0.0f);
    const float t_flip = 4.0f;           /* spool-up + rotation, during which we keep falling */
    float a_net;
    if (n_burn < 1) { n_burn = 1; }
    a_net = (float)n_burn * VEH_T_MAX_EACH * 0.80f / fmaxf(mass, 1.0f) - FSW_G0;
    if (a_net < 0.5f) { a_net = 0.5f; }
    return h_gate + 0.85f * V * t_flip + V * V / (2.0f * a_net) + 120.0f;
}

/* ============================ ZEM/ZEV ============================ */
static void zem_zev(const float r[3], const float v[3], const float rf[3], const float vf[3],
                    const float a[3], float T, float zem[3], float zev[3])
{
    int i;
    for (i = 0; i < 3; ++i) {
        zem[i] = rf[i] - (r[i] + v[i] * T + 0.5f * a[i] * T * T);
        zev[i] = vf[i] - (v[i] + a[i] * T);
    }
}

float pdg_cost(const float r[3], const float v[3], const float rf[3], const float vf[3],
               const float a_drift[3], float gamma, float T)
{
    float zem[3], zev[3];
    zem_zev(r, v, rf, vf, a_drift, T, zem, zev);
    /* min int|u|^2 for the double integrator: d' W^-1 d with W the controllability Gramian */
    return gamma * T + 0.5f * (12.0f / (T * T * T) * v3_dot(zem, zem)
                               - 12.0f / (T * T) * v3_dot(zem, zev)
                               + 4.0f / T * v3_dot(zev, zev));
}

float pdg_optimal_tgo(const float r[3], const float v[3], const float rf[3], const float vf[3],
                      const float a_drift[3], float gamma, float t_lo, float t_hi)
{
    const int NG = 32;
    const float phi = 0.6180340f;
    float best_t = t_lo, best_j = 3.0e38f, a, b, c, d, jc, jd;
    int i;
    for (i = 0; i < NG; ++i) {
        const float T = t_lo + (t_hi - t_lo) * (float)i / (float)(NG - 1);
        const float J = pdg_cost(r, v, rf, vf, a_drift, gamma, T);
        if (J < best_j) { best_j = J; best_t = T; }
    }
    a = fmaxf(t_lo, best_t - (t_hi - t_lo) / (float)(NG - 1));
    b = fminf(t_hi, best_t + (t_hi - t_lo) / (float)(NG - 1));
    c = b - phi * (b - a);
    d = a + phi * (b - a);
    jc = pdg_cost(r, v, rf, vf, a_drift, gamma, c);
    jd = pdg_cost(r, v, rf, vf, a_drift, gamma, d);
    for (i = 0; i < 24; ++i) {
        if (jc < jd) { b = d; d = c; jd = jc; c = b - phi * (b - a); jc = pdg_cost(r, v, rf, vf, a_drift, gamma, c); }
        else         { a = c; c = d; jc = jd; d = a + phi * (b - a); jd = pdg_cost(r, v, rf, vf, a_drift, gamma, d); }
    }
    return 0.5f * (a + b);
}

void pdg_init(pdg_t *g, float gamma, const float r_tgt[3], const float v_tgt[3])
{
    g->gamma = gamma;
    g->tgo = 0.0f;
    g->tgo_freeze = 2.5f;
    v3_copy(g->r_tgt, r_tgt);
    v3_copy(g->v_tgt, v_tgt);
    v3_zero(g->a_cmd);
    g->initialized = 0;
}

void pdg_retarget(pdg_t *g, const float r_tgt[3], const float v_tgt[3])
{
    v3_copy(g->r_tgt, r_tgt);
    v3_copy(g->v_tgt, v_tgt);
    g->initialized = 0;          /* re-optimise tgo from scratch for the new target */
}

void pdg_step(pdg_t *g, const float r[3], const float v[3], const float a_drift[3], float dt)
{
    float zem[3], zev[3], T;
    int i;
    if (!g->initialized) {
        g->tgo = pdg_optimal_tgo(r, v, g->r_tgt, g->v_tgt, a_drift, g->gamma, 1.0f, 90.0f);
        g->initialized = 1;
    } else if (g->tgo > g->tgo_freeze) {
        /* Bellman consistency: on an optimal arc tgo decreases by dt. Re-optimise, but only
         * accept a bounded deviation from the count-down to avoid hopping between minima. */
        const float tc = g->tgo - dt;
        const float to = pdg_optimal_tgo(r, v, g->r_tgt, g->v_tgt, a_drift, g->gamma,
                                         fmaxf(1.0f, tc - 3.0f), tc + 3.0f);
        g->tgo = gnc_clampf(to, tc - 0.1f, tc + 0.1f);
    } else {
        g->tgo -= dt;
    }
    if (g->tgo < 0.3f) { g->tgo = 0.3f; }
    T = g->tgo;
    zem_zev(r, v, g->r_tgt, g->v_tgt, a_drift, T, zem, zev);
    for (i = 0; i < 3; ++i) {
        g->a_cmd[i] = 6.0f * zem[i] / (T * T) - 2.0f * zev[i] / T;
    }
}

int thrust_project(const float F_des[3], float T_min, float T_max, float tilt_max, float F_out[3])
{
    float Fz = F_des[2];
    float Fh = sqrtf(F_des[0] * F_des[0] + F_des[1] * F_des[1]);
    float hx = 0.0f, hy = 0.0f, Fh_max, mag;
    int active = 0;
    if (Fh > 1.0e-3f) { hx = F_des[0] / Fh; hy = F_des[1] / Fh; }
    if (Fz < 1.0e-3f * T_max) { Fz = 1.0e-3f * T_max; active = 1; }     /* cannot pull down */
    if (Fz > T_max) { Fz = T_max; active = 1; }                         /* vertical priority */
    Fh_max = fminf(Fz * tanf(tilt_max), sqrtf(fmaxf(T_max * T_max - Fz * Fz, 0.0f)));
    if (Fh > Fh_max) { Fh = Fh_max; active = 1; }
    mag = sqrtf(Fz * Fz + Fh * Fh);
    if (mag < T_min) {
        /* raise magnitude along the same direction (min throttle) */
        const float s = T_min / fmaxf(mag, 1.0e-3f);
        Fz *= s; Fh *= s;
        active = 1;
    }
    F_out[0] = Fh * hx;
    F_out[1] = Fh * hy;
    F_out[2] = Fz;
    return active;
}

/* ============================ terminal ============================ */
void terminal_accel(const float r[3], const float v[3], const float pad[3], float h_gate,
                    float v_gate, const float a_drift[3], float a_cmd[3])
{
    const float h = r[2] - pad[2];
    const float frac = gnc_clampf(h / h_gate, 0.0f, 1.0f);
    float vref[3], vh;
    int i;
    vref[0] = -TERM_K_P * (r[0] - pad[0]);
    vref[1] = -TERM_K_P * (r[1] - pad[1]);
    vh = sqrtf(vref[0] * vref[0] + vref[1] * vref[1]);
    if (vh > TERM_VH_MAX) { vref[0] *= TERM_VH_MAX / vh; vref[1] *= TERM_VH_MAX / vh; }
    vref[2] = -(TERM_V_TD + (v_gate - TERM_V_TD) * frac);
    for (i = 0; i < 3; ++i) {
        a_cmd[i] = TERM_K_V * (vref[i] - v[i]) - a_drift[i];
    }
}

/* ============================ abort ============================ */
float stopping_margin(float h, float vz, float mass, int32_t n_engines, float h_gate,
                      float a_aero_z)
{
    float a_avail, dh;
    if (vz >= 0.0f) { return 0.0f; }
    a_avail = (float)n_engines * VEH_T_MAX_EACH * 0.95f / fmaxf(mass, 1.0f) - FSW_G0
              + fmaxf(a_aero_z, 0.0f);
    if (a_avail <= 0.01f) { return 99.0f; }
    dh = fmaxf(h - h_gate, 1.0f);
    return (vz * vz / (2.0f * dh)) / a_avail;
}
