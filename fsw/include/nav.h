/**
 * @file nav.h
 * @brief 18-state error-state Kalman filter (GNSS / IMU / radar altimeter).
 *
 * Nominal state: position p_L, velocity v_L, attitude q_LB, gyro bias b_g, accel bias b_a,
 *                GNSS position bias b_n (first-order Gauss-Markov, tau = NAV_GNSS_BIAS_TAU).
 * Error state:   dx = [dp(3) dv(3) dth(3) dbg(3) dba(3) dbn(3)], dth a body-frame (right)
 *                rotation error: R_true = R_hat (I + [dth]x).
 * Modelling the time-correlated GNSS bias keeps the position covariance honest: white-noise
 * R inflation cannot, because 10 Hz updates average white noise but not a slow bias.
 *
 * Propagation is driven by the IMU at the executive rate. Measurement updates are
 * processed as sequential scalar updates (no matrix inverse) after a vector chi-square
 * innovation gate (Cholesky on the <=3x3 innovation covariance). A sensor whose
 * innovations are rejected NAV_FAIL_CONSEC times in a row is latched failed (FDIR).
 */
#ifndef NAV_H
#define NAV_H

#include <stdint.h>

#define NAV_NX 18
#define NAV_IP 0
#define NAV_IV 3
#define NAV_ITH 6
#define NAV_IBG 9
#define NAV_IBA 12
#define NAV_IBN 15

enum { NAV_SENS_GNSS_POS = 0, NAV_SENS_GNSS_VEL, NAV_SENS_RADAR, NAV_N_SENS };

/** Measurement-update result codes. */
enum { NAV_UPD_ACCEPTED = 1, NAV_UPD_REJECTED = 0, NAV_UPD_SENSOR_FAILED = -1, NAV_UPD_INVALID = -2 };

typedef struct {
    int32_t consec_reject;
    int32_t failed;
    int32_t n_accept;
    int32_t n_reject;
    float   last_nis;
} nav_health_t;

typedef struct {
    float p[3], v[3], q[4], bg[3], ba[3], bn[3];
    float P[NAV_NX * NAV_NX];
    float dx[NAV_NX];
    float omega_b[3];        /* bias-corrected angular rate */
    float f_b[3];            /* bias-corrected specific force */
    float t;
    nav_health_t health[NAV_N_SENS];
} nav_t;

typedef struct {
    float pos, vel, att, gyro_bias, accel_bias;   /* initial 1-sigma */
} nav_init_sigma_t;

void nav_init(nav_t *nav, const float p0[3], const float v0[3], const float q0[4],
              const nav_init_sigma_t *sig);
void nav_propagate(nav_t *nav, const float gyro[3], const float accel[3], float dt);
int  nav_update_gnss_pos(nav_t *nav, const float pos[3]);
int  nav_update_gnss_vel(nav_t *nav, const float vel[3]);
/** antenna: 0 = belly (+z_B boresight), 1 = aft (-x_B boresight). Flat ground at z = 0. */
int  nav_update_radar(nav_t *nav, float range, int32_t antenna);

/** Radar measurement model and Jacobian (exposed for unit testing). Returns -1 if the
 *  boresight is too far from nadir. */
int  nav_radar_model(const nav_t *nav, int32_t antenna, float *range_pred, float H[NAV_NX]);

#endif
