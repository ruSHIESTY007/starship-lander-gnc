/** @file aero_model.c  @brief Onboard US76 atmosphere and crossflow aero model. */
#include "aero_model.h"

#include <math.h>

#include "fsw_config.h"
#include "gnc_math.h"

#define R_AIR   287.053f
#define GAMMA_A 1.4f
#define R_E     6356766.0f

void atmo_us76(float h_geom, float *rho, float *a_sound)
{
    static const float hb[4] = { 0.0f, 11000.0f, 20000.0f, 32000.0f };
    static const float Tb[4] = { 288.15f, 216.65f, 216.65f, 228.65f };
    static const float Lb[4] = { -0.0065f, 0.0f, 0.001f, 0.0028f };
    static const float pb[4] = { 101325.0f, 22632.06f, 5474.889f, 868.0187f };
    float h = R_E * h_geom / (R_E + h_geom);
    float T, p;
    int i = 0;
    int k;
    if (h > 47000.0f) { h = 47000.0f; }
    for (k = 1; k < 4; ++k) {
        if (h >= hb[k]) { i = k; }
    }
    T = Tb[i] + Lb[i] * (h - hb[i]);
    if (Lb[i] == 0.0f) {
        p = pb[i] * expf(-FSW_G0 * (h - hb[i]) / (R_AIR * Tb[i]));
    } else {
        p = pb[i] * powf(T / Tb[i], -FSW_G0 / (Lb[i] * R_AIR));
    }
    *rho = p / (R_AIR * T);
    *a_sound = sqrtf(GAMMA_A * R_AIR * T);
}

static float interp(float x, const float *xs, const float *ys, int n)
{
    int i;
    if (x <= xs[0]) { return ys[0]; }
    for (i = 1; i < n; ++i) {
        if (x <= xs[i]) {
            const float f = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
            return ys[i - 1] + f * (ys[i] - ys[i - 1]);
        }
    }
    return ys[n - 1];
}

static const float CDC_M[5] = { 0.0f, 0.4f, 0.9f, 1.2f, 3.0f };
static const float CDC_V[5] = { 1.20f, 1.20f, 1.65f, 1.50f, 1.50f };
static const float CA_M[5]  = { 0.0f, 0.6f, 1.0f, 1.3f, 3.0f };
static const float CA_V[5]  = { 1.00f, 1.00f, 1.60f, 1.40f, 1.30f };

float aero_forces(const float v_air_b[3], const float omega_b[3], float rho, float a_sound,
                  const float flaps[4], float kcd, float F[3], float M[3])
{
    const float u = v_air_b[0], v = v_air_b[1], w = v_air_b[2];
    const float wx = omega_b[0], wy = omega_b[1], wz = omega_b[2];
    const float a_strip = VEH_STRIP_LEN * VEH_DIAMETER;
    const float s_base = 0.25f * GNC_PI * VEH_DIAMETER * VEH_DIAMETER;
    const float ny_mag = sinf(VEH_FLAP_DIHEDRAL), nz = cosf(VEH_FLAP_DIHEDRAL);
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f, Mx = 0.0f, My = 0.0f, Mz = 0.0f;
    float V, ca, Fa;
    int i;
    (void)wx;
    for (i = 0; i < FSW_N_STRIP; ++i) {
        const float xs = VEH_STRIP_X[i];
        const float vy = v + wz * xs;
        const float vz = w - wy * xs;
        const float vc = sqrtf(vy * vy + vz * vz);
        const float k = -0.5f * rho * a_strip * kcd * vc * interp(vc / a_sound, CDC_M, CDC_V, 5);
        Fy += k * vy;
        Fz += k * vz;
        My += -xs * k * vz;
        Mz += xs * k * vy;
    }
    V = sqrtf(u * u + v * v + w * w);
    ca = ((u > 0.0f) ? VEH_CA_NOSE : VEH_CA_TAIL) * interp(V / a_sound, CA_M, CA_V, 5);
    Fa = -0.5f * rho * fabsf(u) * u * s_base * ca;
    Fx += Fa;
    for (i = 0; i < FSW_N_FLAP; ++i) {
        const float xf = VEH_FLAP_X[i], yf = VEH_FLAP_Y[i];
        const float ny = (yf > 0.0f) ? ny_mag : -ny_mag;
        const float lvy = v + wz * xf;
        const float lvz = w + wx * yf - wy * xf;
        const float vn = lvy * ny + lvz * nz;
        const float e = gnc_clampf(flaps[i], 0.0f, 1.0f);
        const float mag = -0.5f * rho * fabsf(vn) * vn * VEH_FLAP_AREA[i] * VEH_FLAP_CF * e;
        const float fy = mag * ny, fz = mag * nz;
        Fy += fy;
        Fz += fz;
        Mx += yf * fz;
        My += -xf * fz;
        Mz += xf * fy;
    }
    F[0] = Fx; F[1] = Fy; F[2] = Fz;
    if (M != 0) { M[0] = Mx; M[1] = My; M[2] = Mz; }
    return 0.5f * rho * V * V;
}

void aero_flap_effectiveness(const float v_air_b[3], float rho, float B[3 * 4], float Fz_per_e[4])
{
    /* moment produced by each flap at unit exposure (no rotation), columns = flaps */
    const float ny_mag = sinf(VEH_FLAP_DIHEDRAL), nz = cosf(VEH_FLAP_DIHEDRAL);
    int i;
    for (i = 0; i < FSW_N_FLAP; ++i) {
        const float xf = VEH_FLAP_X[i], yf = VEH_FLAP_Y[i];
        const float ny = (yf > 0.0f) ? ny_mag : -ny_mag;
        const float vn = v_air_b[1] * ny + v_air_b[2] * nz;
        const float mag = -0.5f * rho * fabsf(vn) * vn * VEH_FLAP_AREA[i] * VEH_FLAP_CF;
        const float fy = mag * ny, fz = mag * nz;
        B[0 * 4 + i] = yf * fz;
        B[1 * 4 + i] = -xf * fz;
        B[2 * 4 + i] = xf * fy;
        Fz_per_e[i] = fz;
    }
}
