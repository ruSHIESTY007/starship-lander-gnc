/**
 * @file guidance.h
 * @brief Guidance laws for every flight phase.
 *
 *  - Glide:   horizontal ZEM/ZEV steering to the ignition aim point, converted to a desired
 *             aerodynamic force direction and inverted *exactly* through crossflow theory into
 *             a body-axis attitude (the normal force on an axisymmetric slender body lies in the
 *             plane of the body axis and the airspeed, perpendicular to the axis).
 *  - Ignition:engine-count-aware stopping-distance trigger ("when to light").
 *  - Landing burn: ZEM/ZEV feedback guidance (Ebrahimi et al. 2008; Guo et al. 2013) with
 *             time-to-go minimising J = Gamma*tgo + 1/2 * int |a_T|^2 dt (closed-form energy for
 *             the double integrator, golden-section search), plus predicted aero/wind
 *             feedforward in the drift term.
 *  - Thrust constraint projection: vertical-priority tilt-cone + magnitude limits.
 *  - Terminal: velocity-profile descent with lateral position hold and wind feedforward.
 *  - Engine-out abort: stopping-margin reachability test.
 */
#ifndef GUIDANCE_H
#define GUIDANCE_H

#include <stdint.h>

/* ---------------- glide ---------------- */
#define GLIDE_TILT_MAX      0.349066f  /* 20 deg max aero-force tilt from vertical */
#define GLIDE_K_ZEM         6.0f       /* ZEM/ZEV gains */
#define GLIDE_K_ZEV         2.0f
#define GLIDE_LEVEL_TIME    2.5f       /* fade steering out over the last 2.5-5 s before the flip */
#define GLIDE_ALPHA_MARGIN  0.349066f  /* AoA kept within 90 +/- 20 deg */
#define GLIDE_TGO_MIN       15.0f      /* floor: bounds the ZEM/ZEV gains late in the glide (slow yaw) */

#define GLIDE_HDG_RATE      0.0873f    /* 5 deg/s heading-line slew limit (continuity reference) */
#define GLIDE_HDG_DEADBAND  0.5f       /* m/s^2: below this, hold heading */

/**
 * Decoupled glide steering. An axisymmetric body can only produce horizontal force along
 * its own (horizontal) axis direction, so cross-heading corrections need a yaw, which is
 * slow. Commanding the full horizontal vector directly leads to a heading-chase spiral.
 * Instead: the along-heading component is realised immediately by pitch, while a
 * rate-limited heading command turns the axis toward the desired force line (either
 * nose or tail, whichever is closer: at most 90 deg).
 *
 * @param r, v       nav position / velocity (L)
 * @param wind       wind estimate at current altitude (L)
 * @param aim        aim point (x, y) for ignition
 * @param h_ign      current ignition altitude estimate
 * @param hdg        in/out heading command (unit horizontal vector, 2D)
 * @param dt         step [s]
 * @param q_des      output attitude target (belly into the airflow)
 * @param a_h_cmd    output achievable horizontal acceleration command (x, y) [m/s^2]
 * @return 0 ok, -1 degenerate (q_des untouched)
 */
int glide_guidance(const float r[3], const float v[3], const float wind[3], const float aim[2],
                   float h_ign, float hdg[2], float dt, float q_des[4], float a_h_cmd[2]);

/** Desired body axis that makes the crossflow normal force point along n_des (unit). */
void glide_axis_for_force(const float v_air[3], const float n_des[3], const float x_cur[3],
                          float x_des[3]);

/* ---------------- ignition ---------------- */
/**
 * Altitude at which to start the flip/landing burn.
 * @param vz         vertical velocity (negative down)
 * @param mass       mass estimate
 * @param n_burn     engines available for the landing burn
 * @param h_gate     terminal gate altitude
 */
float ignition_altitude(float vz, float mass, int32_t n_burn, float h_gate);

/* ---------------- powered descent (ZEM/ZEV) ---------------- */
typedef struct {
    float gamma;            /* time weight in J */
    float tgo;
    float tgo_freeze;       /* below this tgo, count down instead of re-optimising */
    float r_tgt[3], v_tgt[3];
    float a_cmd[3];         /* commanded thrust acceleration (L) */
    int32_t initialized;
} pdg_t;

void  pdg_init(pdg_t *g, float gamma, const float r_tgt[3], const float v_tgt[3]);
void  pdg_retarget(pdg_t *g, const float r_tgt[3], const float v_tgt[3]);
float pdg_cost(const float r[3], const float v[3], const float rf[3], const float vf[3],
               const float a_drift[3], float gamma, float T);
float pdg_optimal_tgo(const float r[3], const float v[3], const float rf[3], const float vf[3],
                      const float a_drift[3], float gamma, float t_lo, float t_hi);
/** a_drift = gravity + predicted aero acceleration. Writes g->a_cmd. */
void  pdg_step(pdg_t *g, const float r[3], const float v[3], const float a_drift[3], float dt);

/**
 * Project a desired thrust force onto the feasible set: angle from vertical <= tilt_max,
 * T_min <= |F| <= T_max, with vertical priority (horizontal authority is sacrificed first).
 * @return 1 if any constraint was active
 */
int thrust_project(const float F_des[3], float T_min, float T_max, float tilt_max, float F_out[3]);

/* ---------------- terminal descent ---------------- */
#define TERM_V_TD       1.0f      /* touchdown descent rate [m/s] */
#define TERM_K_V        1.2f      /* vertical velocity loop gain [1/s] */
#define TERM_K_VH       0.6f      /* lateral velocity loop gain [1/s] (attitude-limited) */
#define TERM_K_P        0.25f     /* lateral position gain [1/s] */
#define TERM_VH_MAX     2.5f

/** hover_ok = 0 (low propellant): no hover-translate, descend where we are. */
void terminal_accel(const float r[3], const float v[3], const float pad[3], float h_gate,
                    float v_gate, const float a_drift[3], int32_t hover_ok, float a_cmd[3]);

/* ---------------- engine-out abort ---------------- */
/**
 * Stopping margin mu = a_required / a_available for nulling the vertical velocity by the
 * gate with n engines at 95 % throttle. mu < 1 feasible; flight rule continues to the pad
 * only if mu < ABORT_MU_PAD.
 */
#define ABORT_MU_PAD 0.80f
float stopping_margin(float h, float vz, float mass, int32_t n_engines, float h_gate,
                      float a_aero_z);

#endif
