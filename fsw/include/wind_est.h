/**
 * @file wind_est.h
 * @brief Aero-model-inverting wind estimator (no air-data sensor on board).
 *
 * State x = [w_x, w_y, k_cd]: horizontal wind in L [m/s] and a crossflow-drag scale factor.
 * Measurement: bias-corrected IMU specific force in body axes,
 *     f_b = (F_aero(R^T (v - w), omega, rho(h), flaps; k_cd) + F_thrust_b) / m,
 * with F_thrust_b reconstructed from chamber-pressure telemetry and the gimbal command.
 * The Jacobian is taken by forward differences of the onboard aero model (3 extra model
 * evaluations). R combines aero-model error, thrust-reconstruction error and IMU noise, so
 * the filter keeps estimating during powered flight where lateral wind loads remain
 * observable against the (well-known) lateral thrust.
 */
#ifndef WIND_EST_H
#define WIND_EST_H

#include <stdint.h>

#include "nav.h"

typedef struct {
    float x[3];
    float P[9];
    float h_ref;        /* altitude at which the estimate was last updated */
    float last_nis;
    int32_t n_update;
    int32_t n_reject;
    int32_t active;     /* last cycle produced an update */
} wind_est_t;

void wind_init(wind_est_t *we, float sigma_w, float sigma_k);
void wind_predict(wind_est_t *we, float dt);
/**
 * @param f_thrust_b   reconstructed thrust force in body axes [N]
 * @param mass         current mass estimate [kg]
 * @return 1 if updated, 0 if rejected/skipped
 */
int  wind_update(wind_est_t *we, const nav_t *nav, const float f_thrust_b[3], float mass,
                 const float flaps[4]);
/** Wind at altitude h extrapolated from the estimate with a 1/7 power-law surface layer. */
void wind_at_altitude(const wind_est_t *we, float h, float w_out[3]);

#endif
