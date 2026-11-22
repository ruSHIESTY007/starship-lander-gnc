/**
 * @file fsw.h
 * @brief Flight software executive: the single entry point called at 100 Hz.
 *
 * Mode sequence:  GLIDE -> FLIP -> BURN -> TERMINAL -> LANDED
 * Abort branches: engine-out before ignition with < 2 engines  -> target backup site (flight rule)
 *                 engine-out in the burn with stopping margin >= ABORT_MU_PAD -> backup site
 *                 no healthy engines                            -> ABORT_NO_THRUST
 *
 * All state is static (no dynamic memory). The interface structs contain only 32-bit
 * fields so they map 1:1 onto Python ctypes structures (sim/fsw_bindings.py).
 */
#ifndef FSW_H
#define FSW_H

#include <stdint.h>

#include "fsw_config.h"
#include "nav.h"

enum {
    MODE_INIT = 0,
    MODE_GLIDE = 1,
    MODE_FLIP = 2,
    MODE_BURN = 3,
    MODE_TERMINAL = 4,
    MODE_LANDED = 5,
    MODE_ABORT_NO_THRUST = 6
};

enum { TARGET_PAD = 0, TARGET_BACKUP = 1 };

/** Powered-descent guidance law selector. */
enum { GUID_ZEMZEV = 0, GUID_NEURAL = 1 };

typedef struct {
    float   t;
    float   gyro[3];
    float   accel[3];
    int32_t gnss_valid;
    float   gnss_pos[3];
    float   gnss_vel[3];
    int32_t radar_valid;
    int32_t radar_antenna;
    float   radar_range;
    float   pc[FSW_N_ENG];
    int32_t eng_ready[FSW_N_ENG];     /* engine controller health discrete (valid while off) */
    int32_t touchdown;
} fsw_input_t;

typedef struct {
    int32_t engine_enable[FSW_N_ENG];
    float   throttle[FSW_N_ENG];
    float   gimbal[2];
    float   flaps[FSW_N_FLAP];
    float   rcs[3];
} fsw_cmd_t;

typedef struct {
    float   p0[3], v0[3], q0[4];      /* navigation handover state */
    float   sig_pos, sig_vel, sig_att;
    float   pad[3], backup[3];
    float   mass0;
    float   gamma;                     /* ZEM/ZEV time weight */
    float   h_gate, v_gate;
    int32_t use_wind_ff;               /* wind estimate used in guidance feedforward */
    int32_t guidance_law;              /* GUID_ZEMZEV / GUID_NEURAL */
} fsw_init_t;

typedef struct {
    float   t;
    int32_t mode;
    int32_t target_id;
    float   nav_p[3], nav_v[3], nav_q[4], nav_bg[3], nav_ba[3];
    float   nav_sig[NAV_NX];          /* sqrt(diag P) */
    float   nis[NAV_N_SENS];
    int32_t sens_failed[NAV_N_SENS];
    float   wind_est[2], wind_sig[2], wind_k;
    float   mass_est;
    float   q_des[4];
    float   att_err[3];
    float   F_des[3];                  /* projected thrust force command (L) */
    float   a_cmd[3];                  /* raw guidance thrust-acceleration command (L) */
    float   a_aero[3];                 /* predicted aero acceleration (L) */
    float   tgo, h_ign, mu_abort;
    float   target[3];
    int32_t n_eng_healthy, n_eng_active;
    int32_t eng_failed[FSW_N_ENG];
    int32_t alloc_saturated;
    float   T_cmd;
    float   tau_req[3];
    int32_t divert_priority;           /* 1: landing not survivable, steering impact to backup */
} fsw_tlm_t;

void fsw_init(const fsw_init_t *init);
void fsw_step(const fsw_input_t *in, fsw_cmd_t *cmd, fsw_tlm_t *tlm);

#endif
