/**
 * @file control.h
 * @brief Attitude control and actuator allocation.
 *
 * Cascaded quaternion controller:
 *   outer loop  w_cmd = sat( K_att * th_err ), with |w_cmd| <= min(w_max, sqrt(2 alpha |th|))
 *               (eigen-axis slew, rate-limited, decelerating within available authority)
 *   inner loop  tau = I (K_rate (w_cmd - w) + integral) + w x I w - M_aero_body_pred
 * Allocation of the required actuator torque, in priority order:
 *   1. engine gimbal (common 2-axis), cancelling engine-out offset torque
 *   2. flaps: bounded pseudo-inverse about a trim exposure, redistributed over the
 *      unsaturated set (cascaded generalised inverse)
 *   3. RCS for the residual (saturating)
 */
#ifndef CONTROL_H
#define CONTROL_H

#include <stdint.h>

typedef struct {
    float k_att;          /* outer-loop gain [1/s]           */
    float w_max;          /* max commanded rate [rad/s]      */
    float alpha_avail;    /* assumed angular accel authority [rad/s^2] */
    float k_rate;         /* inner-loop bandwidth [1/s]      */
    float k_i;            /* integral gain [1/s^2]           */
    float i_lim;          /* integral clamp [rad/s^2]        */
} att_gains_t;

typedef struct {
    float integ[3];
    float th_err[3];
    float w_cmd[3];
    int32_t hold_integrator;
} att_ctrl_t;

void att_ctrl_reset(att_ctrl_t *c);
/** Returns the required body torque (including gyroscopic term), before aero feedforward. */
void att_ctrl_step(att_ctrl_t *c, const att_gains_t *g, const float q_des[4], const float q[4],
                   const float w[3], float dt, float tau_req[3]);

typedef struct {
    float gimbal[2];      /* lateral thrust fractions (y_B, z_B) */
    float flaps[4];
    float rcs[3];
    float tau_ach[3];     /* torque the allocation expects to produce */
    int32_t saturated;
} alloc_t;

/**
 * @param tau_act      torque the actuators must deliver [N m]
 * @param T_total      current total thrust [N] (from chamber pressure)
 * @param tau_eng_off  torque from engine thrust offsets about the CoM (engine-out) [N m]
 * @param v_air_b, rho airflow for flap effectiveness
 * @param flap_trim    baseline exposure the pseudo-inverse is centred on
 * @param use_flaps    0 to command flaps to the trim value only
 */
void alloc_torque(const float tau_act[3], float T_total, const float tau_eng_off[3],
                  const float v_air_b[3], float rho, float flap_trim, int32_t use_flaps,
                  alloc_t *out);

/** Torque about the CoM from each engine's axial thrust (no gimbal). */
void engine_offset_torque(const float T_each[3], float tau[3]);

#endif
