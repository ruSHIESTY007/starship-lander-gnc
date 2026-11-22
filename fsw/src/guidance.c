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
    /* envelope protection: keep the angle of attack within 90 +/- GLIDE_ALPHA_MARGIN.
     * Outside it the aft-heavy body diverges toward tail-first faster than the flaps
     * can recover. */
    if (V > 1.0f) {
        float vh[3], xp[3];
        const float cmax = sinf(GLIDE_ALPHA_MARGIN);
        float ca;
        v3_scale(vh, v_air, 1.0f / V);
        ca = v3_dot(x_des, vh);
        if (fabsf(ca) > cmax) {
            v3_copy(xp, x_des);
            v3_axpy(xp, -ca, vh);
            if (v3_normalize(xp, xp) > 1.0e-6f) {
                v3_scale(x_des, xp, sqrtf(1.0f - cmax * cmax));
                v3_axpy(x_des, gnc_signf(ca) * cmax, vh);
            }
        }
    }
}

int glide_guidance(const float r[3], const float v[3], const float wind[3], const float aim[2],
                   float h_ign, float hdg[2], float dt, float q_des[4], float a_h_cmd[2])
{
    /* Full-vector steering with softened gains: the cross-heading part needs a (slow) yaw,
     * and the nominal ZEM/ZEV gains (6, 2) drive a heading-chase spiral late in the glide.
     * The glide only has to deliver the vehicle inside the landing burn's divert envelope. */
    float t, zem[2], zev[2], ah[2], amag, amax, n[3], vair[3], xd[3], xc[3];
    t = (r[2] - h_ign) / fmaxf(-v[2], 20.0f);
    if (t < GLIDE_TGO_MIN) { t = GLIDE_TGO_MIN; }
    zem[0] = aim[0] - r[0] - v[0] * t;
    zem[1] = aim[1] - r[1] - v[1] * t;
    zev[0] = -v[0];
    zev[1] = -v[1];
    ah[0] = GLIDE_K_ZEM * zem[0] / (t * t) - GLIDE_K_ZEV * zev[0] / t;
    ah[1] = GLIDE_K_ZEM * zem[1] / (t * t) - GLIDE_K_ZEV * zev[1] / t;
    amag = sqrtf(ah[0] * ah[0] + ah[1] * ah[1]);
    amax = FSW_G0 * tanf(GLIDE_TILT_MAX);
    if (amag > amax) { ah[0] *= amax / amag; ah[1] *= amax / amag; }
    {
        /* level the belly before the flip: a nose-up (or -down) start makes the aerodynamic
         * pitch-up throw the vehicle along its heading by up to ~1 g for seconds; starting
         * level makes the flip impulse small and repeatable */
        const float t_ign = (r[2] - h_ign) / fmaxf(-v[2], 20.0f);
        const float lev = gnc_clampf((t_ign - GLIDE_LEVEL_TIME) / GLIDE_LEVEL_TIME, 0.0f, 1.0f);
        ah[0] *= lev; ah[1] *= lev;
    }
    a_h_cmd[0] = ah[0];
    a_h_cmd[1] = ah[1];
    v3_set(n, ah[0], ah[1], FSW_G0);
    (void)v3_normalize(n, n);
    v3_sub(vair, v, wind);
    /* heading continuity: the commanded axis line is the previous one */
    v3_set(xc, hdg[0], hdg[1], 0.0f);
    glide_axis_for_force(vair, n, xc, xd);
    {
        const float hn = sqrtf(xd[0] * xd[0] + xd[1] * xd[1]);
        if (hn > 0.2f) {
            /* store the new axis line, rate-limited to GLIDE_HDG_RATE */
            float d0 = xd[0] / hn, d1 = xd[1] / hn, cr, dp, step, c, s;
            if (d0 * hdg[0] + d1 * hdg[1] < 0.0f) { d0 = -d0; d1 = -d1; }
            cr = hdg[0] * d1 - hdg[1] * d0;
            dp = hdg[0] * d0 + hdg[1] * d1;
            step = gnc_clampf(atan2f(cr, dp), -GLIDE_HDG_RATE * dt, GLIDE_HDG_RATE * dt);
            c = cosf(step); s = sinf(step);
            d0 = c * hdg[0] - s * hdg[1];
            d1 = s * hdg[0] + c * hdg[1];
            hdg[0] = d0; hdg[1] = d1;
        }
    }
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
    return h_gate + 0.85f * V * t_flip + V * V / (2.0f * a_net) + 200.0f;
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
    g->tgo_freeze = 3.0f;
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
    /* upper bound on tgo: forbid hover-and-divert (or climbing) solutions, which are
     * energy-optimal for large lateral errors but waste propellant at 1 g */
    const float t_hi = gnc_clampf(2.5f * (r[2] - g->r_tgt[2]) / fmaxf(-v[2], 15.0f) + 2.0f, 4.0f, 60.0f);
    if (!g->initialized) {
        g->tgo = pdg_optimal_tgo(r, v, g->r_tgt, g->v_tgt, a_drift, g->gamma, 1.0f, t_hi);
        g->initialized = 1;
    } else if (g->tgo > g->tgo_freeze) {
        /* Bellman consistency: on an optimal arc tgo decreases by dt. Re-optimise, but only
         * accept a bounded deviation from the count-down to avoid hopping between minima. */
        const float tc = g->tgo - dt;
        const float to = pdg_optimal_tgo(r, v, g->r_tgt, g->v_tgt, a_drift, g->gamma,
                                         fmaxf(1.0f, tc - 3.0f), fminf(tc + 3.0f, fmaxf(t_hi, tc)));
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
                    float v_gate, const float a_drift[3], int32_t hover_ok, float a_cmd[3])
{
    const float h = r[2] - pad[2];
    const float frac = gnc_clampf(h / h_gate, 0.0f, 1.0f);
    float vref[3], vh;
    int i;
    float eh, vh_max, slow;
    vref[0] = -TERM_K_P * (r[0] - pad[0]);
    vref[1] = -TERM_K_P * (r[1] - pad[1]);
    /* lateral speed limit tapers near the ground so touchdown is (nearly) vertical */
    vh_max = TERM_VH_MAX * gnc_clampf(h / 12.0f, 0.2f, 1.0f);
    vh = sqrtf(vref[0] * vref[0] + vref[1] * vref[1]);
    if (vh > vh_max) { vref[0] *= vh_max / vh; vref[1] *= vh_max / vh; }
    /* hover-translate: slow the descent while far from the pad (costs propellant, buys accuracy) */
    eh = sqrtf((r[0] - pad[0]) * (r[0] - pad[0]) + (r[1] - pad[1]) * (r[1] - pad[1]));
    slow = (h > 8.0f && hover_ok) ? gnc_clampf(1.4f - eh / 10.0f, 0.3f, 1.0f) : 1.0f;
    vref[2] = -(TERM_V_TD + (v_gate - TERM_V_TD) * frac) * slow;
    for (i = 0; i < 3; ++i) {
        const float k = (i == 2) ? TERM_K_V : TERM_K_VH;
        a_cmd[i] = k * (vref[i] - v[i]) - a_drift[i];
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
