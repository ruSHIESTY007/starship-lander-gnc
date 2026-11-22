/**
 * @file nn_policy.c
 * @brief MLP guidance policy inference.
 *
 * Parameter blob layout (row-major weights, y = W x + b):
 *   W1[NN_H][NN_IN], b1[NN_H], W2[NN_H][NN_H], b2[NN_H], W3[NN_OUT][NN_H], b3[NN_OUT],
 *   x_mean[NN_IN], x_std[NN_IN], y_mean[NN_OUT], y_std[NN_OUT]
 */
#include "nn_policy.h"

#include <math.h>

static float s_p[NN_N_PARAMS];
static int32_t s_loaded = 0;

#define OFF_W1 0
#define OFF_B1 (OFF_W1 + NN_H * NN_IN)
#define OFF_W2 (OFF_B1 + NN_H)
#define OFF_B2 (OFF_W2 + NN_H * NN_H)
#define OFF_W3 (OFF_B2 + NN_H)
#define OFF_B3 (OFF_W3 + NN_OUT * NN_H)
#define OFF_XM (OFF_B3 + NN_OUT)
#define OFF_XS (OFF_XM + NN_IN)
#define OFF_YM (OFF_XS + NN_IN)
#define OFF_YS (OFF_YM + NN_OUT)

int32_t nn_policy_load(const float *params, int32_t n)
{
    int32_t i;
    if (n != NN_N_PARAMS) { return -1; }
    for (i = 0; i < n; ++i) {
        if (!(params[i] == params[i])) { return -2; }      /* reject NaN */
        s_p[i] = params[i];
    }
    for (i = 0; i < NN_IN; ++i) { if (s_p[OFF_XS + i] <= 0.0f) { return -3; } }
    s_loaded = 1;
    return 0;
}

int32_t nn_policy_loaded(void) { return s_loaded; }
void nn_policy_unload(void) { s_loaded = 0; }

void nn_policy_eval(const float x[NN_IN], float y[NN_OUT])
{
    float xn[NN_IN], h1[NN_H], h2[NN_H];
    int i, j;
    for (i = 0; i < NN_IN; ++i) { xn[i] = (x[i] - s_p[OFF_XM + i]) / s_p[OFF_XS + i]; }
    for (i = 0; i < NN_H; ++i) {
        float acc = s_p[OFF_B1 + i];
        for (j = 0; j < NN_IN; ++j) { acc += s_p[OFF_W1 + i * NN_IN + j] * xn[j]; }
        h1[i] = tanhf(acc);
    }
    for (i = 0; i < NN_H; ++i) {
        float acc = s_p[OFF_B2 + i];
        for (j = 0; j < NN_H; ++j) { acc += s_p[OFF_W2 + i * NN_H + j] * h1[j]; }
        h2[i] = tanhf(acc);
    }
    for (i = 0; i < NN_OUT; ++i) {
        float acc = s_p[OFF_B3 + i];
        for (j = 0; j < NN_H; ++j) { acc += s_p[OFF_W3 + i * NN_H + j] * h2[j]; }
        y[i] = acc * s_p[OFF_YS + i] + s_p[OFF_YM + i];
    }
}

int32_t nn_policy_accel(const float r_rel[3], const float v[3], const float a_drift[3],
                        float mass, float a_cmd[3])
{
    float x[NN_IN];
    if (!s_loaded) { return -1; }
    x[0] = r_rel[0]; x[1] = r_rel[1]; x[2] = r_rel[2];
    x[3] = v[0]; x[4] = v[1]; x[5] = v[2];
    x[6] = a_drift[2] + 9.80665f;      /* predicted vertical aero acceleration */
    x[7] = mass * 1.0e-5f;
    nn_policy_eval(x, a_cmd);
    return 0;
}
