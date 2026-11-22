/**
 * @file nn_policy.h
 * @brief Learned powered-descent guidance policy: fixed-size MLP inference, static weights.
 *
 * Input  (NN_IN = 8):  r - r_target (3), v (3), predicted vertical aero accel, mass / 1e5
 * Output (NN_OUT = 3): commanded thrust acceleration in L [m/s^2]
 * Architecture: 8 -> 64 tanh -> 64 tanh -> 3 linear, with input/output affine normalisation.
 * Weights are produced offline by imitation learning on convex-optimal (G-FOLD) solutions
 * with DAgger (analysis/train_policy.py) and uploaded via nn_policy_load().
 * No allocation, no recursion; worst-case cost is a fixed 4.8k multiply-adds.
 */
#ifndef NN_POLICY_H
#define NN_POLICY_H

#include <stdint.h>

#define NN_IN  8
#define NN_H   64
#define NN_OUT 3
#define NN_N_PARAMS (NN_H * NN_IN + NN_H + NN_H * NN_H + NN_H + NN_OUT * NN_H + NN_OUT \
                     + 2 * NN_IN + 2 * NN_OUT)

/** Load a flat parameter blob (layout documented in nn_policy.c). Returns 0 on success. */
int32_t nn_policy_load(const float *params, int32_t n);
int32_t nn_policy_loaded(void);
void    nn_policy_unload(void);
/** Raw network evaluation on an 8-vector input (already assembled). */
void    nn_policy_eval(const float x[NN_IN], float y[NN_OUT]);
/** Guidance call. Returns 0 if a command was produced, -1 if no policy is loaded. */
int32_t nn_policy_accel(const float r_rel[3], const float v[3], const float a_drift[3],
                        float mass, float a_cmd[3]);

#endif
