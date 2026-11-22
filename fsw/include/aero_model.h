/**
 * @file aero_model.h
 * @brief Onboard atmosphere (US76) and aerodynamic model used for wind estimation and
 *        guidance feedforward. Mirrors sim/atmosphere.py and sim/aero.py at nominal values.
 */
#ifndef AERO_MODEL_H
#define AERO_MODEL_H

/** US Standard Atmosphere 1976, geometric altitude h [m] -> density, speed of sound. */
void atmo_us76(float h, float *rho, float *a_sound);

/**
 * Aerodynamic force/moment in the body frame about the design CoM.
 * @param v_air_b   vehicle velocity relative to the air mass, body frame [m/s]
 * @param omega_b   body rate [rad/s]
 * @param flaps     flap exposures [0,1] (FR, FL, AR, AL)
 * @param kcd       crossflow drag scale factor (1 = nominal)
 * @param F, M      outputs [N], [N m]; M may be NULL
 * @return dynamic pressure [Pa]
 */
float aero_forces(const float v_air_b[3], const float omega_b[3], float rho, float a_sound,
                  const float flaps[4], float kcd, float F[3], float M[3]);

/** Flap force-per-exposure and moment-per-exposure for allocation (unit exposure). */
void aero_flap_effectiveness(const float v_air_b[3], float rho, float B[3 * 4], float Fz_per_e[4]);

#endif
