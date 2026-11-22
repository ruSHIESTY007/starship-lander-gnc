/**
 * @file fsw.c
 * @brief Flight software executive: sensor processing, engine FDIR, mode management,
 *        guidance dispatch, attitude control, allocation and engine management.
 */
#include "fsw.h"

#include <math.h>

#include "aero_model.h"
#include "control.h"
#include "fsw_config.h"
#include "gnc_math.h"
#include "guidance.h"
#include "nav.h"
#include "nn_policy.h"
#include "wind_est.h"

/* ---------------- engine FDIR / management constants ---------------- */
#define ENG_PC_FAIL_RATIO     0.5f    /* measured/expected Pc below this -> anomaly   */
#define ENG_PC_FAIL_CYCLES    3       /* consecutive anomalous cycles to declare out  */
#define ENG_SHUTDOWN_DWELL    0.5f
#define BURN_TILT_MAX         0.436332f   /* 25 deg */
#define TERM_TILT_MAX         0.174533f   /* 10 deg above TERM_TILT_TAPER_H */
#define TERM_TILT_MIN         0.052360f   /* 3 deg at touchdown */
#define TERM_TILT_TAPER_H     15.0f
#define TGO_HANDOVER          2.0f        /* ZEM/ZEV -> terminal before its gains go singular */
#define ENG_SINGLE_FRAC       0.90f       /* drop to one engine once demand fits in 90 % of one */
#define ABORT_MU_UNSURVIVABLE 1.10f       /* stopping margin beyond which touchdown is not survivable */
#define DIVERT_TILT           0.610865f   /* 35 deg: lateral-priority thrust tilt toward the backup */
#define FLIP_PRE_TIME         4.0f        /* max aerodynamic pitch-up time before ignition [s] */
#define FLIP_IGNITE_ERR       0.610865f   /* ignite once within 35 deg of the thrust attitude */
#define FLIP_DISP             25.0f       /* backward flip displacement along heading (flip + its velocity) [m] */
#define NN_HANDOVER_T         3.0f        /* s to gate: learned policy -> ZEM/ZEV */
#define FLIP_WATCHDOG_T       7.0f
#define FLIP_WATCHDOG_H       250.0f
#define FLIP_PRE_TIME_EST     3.0f        /* expected pre-flip duration used in the trigger */
#define FLIP_DONE_ERR         0.139626f   /* 8 deg */
#define WIND_AIM_LEAD         3.0f        /* s of downwind drift pre-compensated in the aim */
#define H_TD                  (-VEH_LEG_X)  /* CoM height above ground at touchdown */

typedef struct {
    fsw_init_t cfg;
    nav_t nav;
    wind_est_t wind;
    pdg_t pdg;
    att_ctrl_t att;
    int32_t mode;
    int32_t target_id;
    float target[3];
    float target_cg[3];       /* target with the touchdown CoM height added */
    float mass;
    float t;
    float t_mode;
    int32_t eng_on[FSW_N_ENG];
    float eng_t_on[FSW_N_ENG];
    int32_t eng_failed[FSW_N_ENG];
    int32_t eng_anom[FSW_N_ENG];
    float throttle_cmd;
    float gimbal_cmd[2];
    float gimbal_trim[2];     /* low-passed gimbal: steady thrust-line offset */
    float flaps_cmd[FSW_N_FLAP];
    float shutdown_timer;
    float heading_flip[3];
    float glide_hdg[2];       /* glide heading command (unit, horizontal) */
    float h_ign;
    float mu_abort;
    int32_t divert_priority;
    int32_t frame;
    int32_t nn_handed_over;   /* learned policy -> ZEM/ZEV handover latched */
    int32_t burn_single;      /* latched: burn continues on one engine (2-engine floor > demand) */
    float   single_timer;
} fsw_state_t;

static fsw_state_t S;

static const att_gains_t GAINS_GLIDE = { 0.6f, 0.08f, 0.03f, 2.0f, 0.2f, 0.02f };
static const att_gains_t GAINS_FLIP  = { 1.2f, 0.45f, 0.25f, 4.0f, 0.5f, 0.05f };
static const att_gains_t GAINS_BURN  = { 0.8f, 0.15f, 0.15f, 3.5f, 0.3f, 0.03f };

/* ------------------------------------------------------------------ */
static int32_t count_healthy(void)
{
    int32_t i, n = 0;
    for (i = 0; i < FSW_N_ENG; ++i) { if (!S.eng_failed[i]) { ++n; } }
    return n;
}

static int32_t count_active(void)
{
    int32_t i, n = 0;
    for (i = 0; i < FSW_N_ENG; ++i) { if (S.eng_on[i] && !S.eng_failed[i]) { ++n; } }
    return n;
}

static void set_target(int32_t id)
{
    const float *src = (id == TARGET_PAD) ? S.cfg.pad : S.cfg.backup;
    float rf[3], vf[3];
    S.target_id = id;
    v3_copy(S.target, src);
    v3_copy(S.target_cg, src);
    S.target_cg[2] += H_TD;
    rf[0] = src[0]; rf[1] = src[1]; rf[2] = src[2] + H_TD + S.cfg.h_gate;
    v3_set(vf, 0.0f, 0.0f, -S.cfg.v_gate);
    pdg_retarget(&S.pdg, rf, vf);
}

static void enter_mode(int32_t m)
{
    S.mode = m;
    S.t_mode = 0.0f;
    S.att.hold_integrator = 0;
}

void fsw_init(const fsw_init_t *init)
{
    nav_init_sigma_t sig;
    float rf[3], vf[3];
    int i;
    S.cfg = *init;
    sig.pos = init->sig_pos;
    sig.vel = init->sig_vel;
    sig.att = init->sig_att;
    sig.gyro_bias = 1.0e-5f;
    sig.accel_bias = 0.015f;
    nav_init(&S.nav, init->p0, init->v0, init->q0, &sig);
    wind_init(&S.wind, 12.0f, 0.15f);
    rf[0] = init->pad[0]; rf[1] = init->pad[1]; rf[2] = init->pad[2] + H_TD + init->h_gate;
    v3_set(vf, 0.0f, 0.0f, -init->v_gate);
    pdg_init(&S.pdg, init->gamma, rf, vf);
    att_ctrl_reset(&S.att);
    S.target_id = TARGET_PAD;
    v3_copy(S.target, init->pad);
    v3_copy(S.target_cg, init->pad);
    S.target_cg[2] += H_TD;
    S.mass = init->mass0;
    S.t = 0.0f;
    S.frame = 0;
    for (i = 0; i < FSW_N_ENG; ++i) {
        S.eng_on[i] = 0; S.eng_t_on[i] = 0.0f; S.eng_failed[i] = 0; S.eng_anom[i] = 0;
    }
    for (i = 0; i < FSW_N_FLAP; ++i) { S.flaps_cmd[i] = 0.5f; }
    S.throttle_cmd = VEH_THROTTLE_MIN;
    S.gimbal_cmd[0] = 0.0f; S.gimbal_cmd[1] = 0.0f;
    S.gimbal_trim[0] = 0.0f; S.gimbal_trim[1] = 0.0f;
    S.shutdown_timer = 0.0f;
    v3_set(S.heading_flip, 1.0f, 0.0f, 0.0f);
    {
        /* initialise the glide heading command from the handover nose direction */
        float R0[9], hn;
        q_to_dcm(R0, init->q0);
        hn = sqrtf(R0[0] * R0[0] + R0[3] * R0[3]);
        S.glide_hdg[0] = (hn > 1e-3f) ? R0[0] / hn : 1.0f;
        S.glide_hdg[1] = (hn > 1e-3f) ? R0[3] / hn : 0.0f;
    }
    S.h_ign = 0.0f;
    S.mu_abort = 0.0f;
    S.divert_priority = 0;
    S.burn_single = 0;
    S.nn_handed_over = 0;
    S.single_timer = 0.0f;
    enter_mode(MODE_GLIDE);
}

/* ---------------- engine FDIR: chamber pressure vs. expected ---------------- */
static int32_t engine_fdir(const float pc[FSW_N_ENG], const int32_t ready[FSW_N_ENG])
{
    int32_t i, newly = 0;
    for (i = 0; i < FSW_N_ENG; ++i) {
        float expected;
        if (S.eng_failed[i]) { continue; }
        if (!S.eng_on[i]) {
            /* engine off: rely on the controller's readiness discrete (debounced) */
            S.eng_anom[i] = ready[i] ? 0 : S.eng_anom[i] + 1;
            if (S.eng_anom[i] >= ENG_PC_FAIL_CYCLES) { S.eng_failed[i] = 1; newly = 1; }
            continue;
        }
        if (S.eng_t_on[i] < VEH_SPOOL_UP_TIME + 0.3f) {
            /* during start: thrust ramp should be at least half the nominal ramp */
            expected = fminf(S.eng_t_on[i] / VEH_SPOOL_UP_TIME, S.throttle_cmd);
            if (S.eng_t_on[i] < 0.5f) { continue; }
        } else {
            expected = S.throttle_cmd;
        }
        if (pc[i] < ENG_PC_FAIL_RATIO * expected) {
            if (++S.eng_anom[i] >= ENG_PC_FAIL_CYCLES) {
                S.eng_failed[i] = 1;
                S.eng_on[i] = 0;
                newly = 1;
            }
        } else {
            S.eng_anom[i] = 0;
        }
    }
    return newly;
}

/**
 * Engine-set management. The mode sets the engine count (flip: all, burn: 2, terminal: 1);
 * the set is chosen from a preference list that minimises the thrust-line offset from the
 * centreline (centre engine first), using healthy engines only. Engines not in the set are
 * shut down after a dwell; engines in the set that are off are (re)lit -- after an
 * engine-out this swaps a healthy engine in.
 */
static void engine_manage(float F_mag, int32_t n_target, float dt)
{
    static const int8_t PREF2[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
    static const int8_t PREF1[3] = { 0, 1, 2 };
    int32_t want[FSW_N_ENG] = { 0, 0, 0 };
    int32_t i, n_set = 0;
    (void)F_mag;
    if (n_target >= 3) {
        for (i = 0; i < FSW_N_ENG; ++i) { want[i] = !S.eng_failed[i]; }
    } else if (n_target == 2) {
        for (i = 0; i < 3; ++i) {
            if (!S.eng_failed[PREF2[i][0]] && !S.eng_failed[PREF2[i][1]]) {
                want[PREF2[i][0]] = 1; want[PREF2[i][1]] = 1; n_set = 2; break;
            }
        }
        if (n_set == 0) { n_target = 1; }
    }
    if (n_target == 1) {
        for (i = 0; i < 3; ++i) {
            if (!S.eng_failed[PREF1[i]]) { want[PREF1[i]] = 1; break; }
        }
    }
    /* relight immediately (engine swap after a failure), shut down after a dwell */
    for (i = 0; i < FSW_N_ENG; ++i) {
        if (want[i] && !S.eng_on[i]) { S.eng_on[i] = 1; S.eng_t_on[i] = 0.0f; }
    }
    {
        int32_t extra = 0;
        for (i = 0; i < FSW_N_ENG; ++i) { if (S.eng_on[i] && !want[i]) { extra = 1; } }
        S.shutdown_timer = extra ? S.shutdown_timer + dt : 0.0f;
        if (S.shutdown_timer > ENG_SHUTDOWN_DWELL) {
            for (i = 0; i < FSW_N_ENG; ++i) { if (!want[i]) { S.eng_on[i] = 0; } }
            S.shutdown_timer = 0.0f;
        }
    }
}

/** Thrust available right now from running engines (spooling engines count partially). */
static void thrust_bounds(float *Tmin, float *Tmax)
{
    float tmax = 0.0f;
    int n = 0, i;
    for (i = 0; i < FSW_N_ENG; ++i) {
        if (S.eng_on[i] && !S.eng_failed[i]) {
            tmax += VEH_T_MAX_EACH * fminf(1.0f, fmaxf(S.eng_t_on[i] / VEH_SPOOL_UP_TIME, 0.0f));
            ++n;
        }
    }
    *Tmax = fmaxf(tmax, 1.0f);
    *Tmin = fminf((float)n * VEH_THROTTLE_MIN * VEH_T_MAX_EACH, *Tmax);
}

/**
 * Attitude that points the *thrust vector* (body axis deflected by the steady gimbal trim,
 * which with an off-centre engine set is several degrees) along u, belly toward `heading`.
 * Writes the body-frame thrust direction to d_b.
 */
static void thrust_attitude(const float u[3], const float heading[3], float q_des[4], float d_b[3])
{
    float q0[4], ex[3] = { 1.0f, 0.0f, 0.0f }, ax[3], qr[4], th[3];
    float s, c;
    const float gy = S.gimbal_trim[0], gz = S.gimbal_trim[1];
    v3_set(d_b, sqrtf(fmaxf(1.0f - gy * gy - gz * gz, 0.0f)), gy, gz);
    if (att_from_two_axes(q0, 0, 1.0f, u, 2, heading) != 0) { return; }
    /* Q rotates d_b onto e_x so that R0 Q d_b = R0 e_x = u */
    v3_cross(ax, d_b, ex);
    s = v3_normalize(ax, ax);
    c = v3_dot(d_b, ex);
    v3_scale(th, ax, atan2f(s, c));
    q_from_rotvec(qr, th);
    q_mul(q_des, q0, qr);
    (void)q_normalize(q_des);
}

/** Powered guidance acceleration command (L) for the current target. */
static void powered_guidance(const float a_drift[3], float dt, float a_cmd[3])
{
    /* ZEM/ZEV always runs: it also provides the time-to-go used for the terminal handover */
    pdg_step(&S.pdg, S.nav.p, S.nav.v, a_drift, dt);
    v3_copy(a_cmd, S.pdg.a_cmd);
    if (S.cfg.guidance_law == GUID_NEURAL && !S.nn_handed_over) {
        /* learned policy shapes the burn; a runtime monitor hands over to the verified
         * feedback law for the final seconds (or if the policy stops descending) */
        float r_rel[3], a_nn[3];
        const float h = S.nav.p[2] - S.pdg.r_tgt[2];
        if (h / fmaxf(-S.nav.v[2], 1e-3f) < NN_HANDOVER_T || (S.nav.v[2] > -1.0f && h > 5.0f)) {
            S.nn_handed_over = 1;
            S.pdg.initialized = 0;      /* fresh optimal tgo from the actual state */
            pdg_step(&S.pdg, S.nav.p, S.nav.v, a_drift, dt);
            v3_copy(a_cmd, S.pdg.a_cmd);
            return;
        }
        v3_sub(r_rel, S.nav.p, S.pdg.r_tgt);
        if (nn_policy_accel(r_rel, S.nav.v, a_drift, S.mass, a_nn) == 0) { v3_copy(a_cmd, a_nn); }
    }
}

void fsw_step(const fsw_input_t *in, fsw_cmd_t *cmd, fsw_tlm_t *tlm)
{
    const float dt = FSW_DT;
    const float g_vec[3] = { 0.0f, 0.0f, -FSW_G0 };
    float R[9], x_b[3], T_each[FSW_N_ENG], T_total = 0.0f, F_thr_b[3], tau_off[3];
    float rho, a_snd, wind_L[3], vair_L[3], vair_b[3], F_aero_b[3], M_body[3], a_aero_L[3];
    float q_des[4], F_des[3] = { 0.0f, 0.0f, 0.0f }, a_cmd[3] = { 0.0f, 0.0f, 0.0f };
    float tau_req[3], tau_act[3], T_cmd = 0.0f, flap_trim = 0.5f;
    const att_gains_t *gains = &GAINS_GLIDE;
    int32_t use_flaps = 1, newly_failed;
    alloc_t al;
    int i;
    const float zero_flaps[FSW_N_FLAP] = { 0.0f, 0.0f, 0.0f, 0.0f };

    S.t = in->t;
    S.t_mode += dt;
    S.frame++;

    /* ================= navigation ================= */
    nav_propagate(&S.nav, in->gyro, in->accel, dt);
    if (in->gnss_valid) {
        (void)nav_update_gnss_pos(&S.nav, in->gnss_pos);
        (void)nav_update_gnss_vel(&S.nav, in->gnss_vel);
    }
    if (in->radar_valid) { (void)nav_update_radar(&S.nav, in->radar_range, in->radar_antenna); }
    q_to_dcm(R, S.nav.q);
    v3_set(x_b, R[0], R[3], R[6]);

    /* ================= engines: FDIR, thrust reconstruction, mass ================= */
    for (i = 0; i < FSW_N_ENG; ++i) {
        if (S.eng_on[i]) { S.eng_t_on[i] += dt; }
    }
    newly_failed = engine_fdir(in->pc, in->eng_ready);
    for (i = 0; i < FSW_N_ENG; ++i) {
        T_each[i] = S.eng_failed[i] ? 0.0f : fmaxf(in->pc[i], 0.0f) * VEH_T_MAX_EACH;
        if (T_each[i] < 0.02f * VEH_T_MAX_EACH) { T_each[i] = 0.0f; }
        T_total += T_each[i];
    }
    {
        const float gy = S.gimbal_cmd[0], gz = S.gimbal_cmd[1];
        v3_set(F_thr_b, T_total * sqrtf(fmaxf(1.0f - gy * gy - gz * gz, 0.0f)), T_total * gy, T_total * gz);
    }
    engine_offset_torque(T_each, tau_off);
    {
        /* Geometric gimbal trim that puts the thrust line through the CoM (feedforward from
         * chamber pressure, independent of the control loop), lightly low-passed. */
        float gy = 0.0f, gz = 0.0f;
        const float a = dt / 0.3f;
        if (T_total > 0.05f * VEH_T_MAX_EACH) {
            gy = gnc_clampf(-tau_off[2] / (VEH_X_AFT * T_total), -VEH_GIMBAL_MAX, VEH_GIMBAL_MAX);
            gz = gnc_clampf(-tau_off[1] / (-VEH_X_AFT * T_total), -VEH_GIMBAL_MAX, VEH_GIMBAL_MAX);
        }
        S.gimbal_trim[0] += a * (gy - S.gimbal_trim[0]);
        S.gimbal_trim[1] += a * (gz - S.gimbal_trim[1]);
    }
    S.mass -= T_total / (VEH_ISP * FSW_G0) * dt;
    if (S.mass < VEH_M_DRY) { S.mass = VEH_M_DRY; }

    /* ================= wind estimation (20 Hz) ================= */
    wind_predict(&S.wind, dt);
    if (S.frame % 5 == 0 && S.mode != MODE_LANDED) {
        (void)wind_update(&S.wind, &S.nav, F_thr_b, S.mass, S.flaps_cmd);
    }

    /* ================= aero prediction (feedforward) ================= */
    atmo_us76(S.nav.p[2], &rho, &a_snd);
    if (S.cfg.use_wind_ff) {
        wind_at_altitude(&S.wind, S.nav.p[2], wind_L);
    } else {
        v3_zero(wind_L);
    }
    v3_sub(vair_L, S.nav.v, wind_L);
    q_rotate_inv(vair_b, S.nav.q, vair_L);
    (void)aero_forces(vair_b, S.nav.omega_b, rho, a_snd, zero_flaps,
                      S.cfg.use_wind_ff ? S.wind.x[2] : 1.0f, F_aero_b, M_body);
    q_rotate(a_aero_L, S.nav.q, F_aero_b);
    v3_scale(a_aero_L, a_aero_L, 1.0f / S.mass);

    q_copy(q_des, S.nav.q);

    /* ================= mode logic & guidance ================= */
    switch (S.mode) {
    case MODE_GLIDE: {
        const int32_t n_h = count_healthy();
        const int32_t n_burn = (n_h >= 2) ? 2 : n_h;
        float aim[2], a_h[2];
        if (n_h < 2 && S.target_id == TARGET_PAD) { set_target(TARGET_BACKUP); }   /* flight rule */
        if (n_h == 0) { enter_mode(MODE_ABORT_NO_THRUST); break; }
        S.h_ign = S.target_cg[2] + ignition_altitude(S.nav.v[2], S.mass, n_burn, S.cfg.h_gate);
        /* aim point: pre-compensate the (repeatable) aerodynamic flip impulse, which pushes
         * the vehicle backward along its heading, and the downwind drift */
        aim[0] = S.target_cg[0] + FLIP_DISP * S.glide_hdg[0]
                 - (S.cfg.use_wind_ff ? WIND_AIM_LEAD * wind_L[0] : 0.0f);
        aim[1] = S.target_cg[1] + FLIP_DISP * S.glide_hdg[1]
                 - (S.cfg.use_wind_ff ? WIND_AIM_LEAD * wind_L[1] : 0.0f);
        {
            float wg[3];
            wind_at_altitude(&S.wind, S.nav.p[2], wg);   /* glide geometry always uses the estimate */
            (void)glide_guidance(S.nav.p, S.nav.v, wg, aim, S.h_ign, S.glide_hdg, dt, q_des, a_h);
        }
        gains = &GAINS_GLIDE;
        flap_trim = 0.5f;
        S.h_ign += FLIP_PRE_TIME_EST * fmaxf(-S.nav.v[2], 0.0f);
        if (S.nav.p[2] <= S.h_ign) {
            float hz[3];
            v3_set(hz, x_b[0], x_b[1], 0.0f);
            if (v3_normalize(hz, hz) < 1e-3f) { v3_set(hz, 1.0f, 0.0f, 0.0f); }
            v3_copy(S.heading_flip, hz);
            S.pdg.initialized = 0;
            enter_mode(MODE_FLIP);    /* aerodynamic pitch-up starts; engines follow */
        }
        break;
    }
    case MODE_FLIP:
    case MODE_BURN: {
        float a_drift[3], Tmin, Tmax, u[3], th[3], d_b[3], d_L[3];
        v3_add(a_drift, g_vec, a_aero_L);
        powered_guidance(a_drift, dt, a_cmd);
        v3_scale(F_des, a_cmd, S.mass);
        if (newly_failed && S.mode == MODE_BURN) {
            const int32_t n_h = count_healthy();
            S.mu_abort = stopping_margin(S.nav.p[2] - S.target_cg[2], S.nav.v[2], S.mass, n_h,
                                         S.cfg.h_gate, a_aero_L[2]);
            if ((n_h == 0 || S.mu_abort >= ABORT_MU_PAD) && S.target_id == TARGET_PAD) {
                set_target(TARGET_BACKUP);
            }
            /* a non-survivable touchdown: maximise distance from the pad instead */
            if (S.mu_abort >= ABORT_MU_UNSURVIVABLE) { S.divert_priority = 1; }
        }
        if (count_healthy() == 0) { enter_mode(MODE_ABORT_NO_THRUST); break; }
        if (S.mode == MODE_BURN) {
            /* two engines cannot throttle below 2*Tmin: once the demand stays under that
             * floor (light vehicle, end of burn) continue on one engine -- latched, no relight */
            if (v3_norm(F_des) < 2.0f * VEH_THROTTLE_MIN * VEH_T_MAX_EACH * 1.02f) { S.single_timer += dt; }
            else { S.single_timer = 0.0f; }
            if (S.single_timer > 0.5f) { S.burn_single = 1; }
            engine_manage(v3_norm(F_des), S.burn_single ? 1 : 2, dt);
        }
        thrust_bounds(&Tmin, &Tmax);
        if (S.divert_priority) {
            /* lateral-priority: full thrust tilted toward the backup site */
            /* steer directly away from the pad (toward the backup if still overhead) */
            float dh[3];
            v3_set(dh, S.nav.p[0] - S.cfg.pad[0], S.nav.p[1] - S.cfg.pad[1], 0.0f);
            if (v3_norm(dh) < 5.0f) {
                v3_set(dh, S.cfg.backup[0] - S.nav.p[0], S.cfg.backup[1] - S.nav.p[1], 0.0f);
            }
            (void)v3_normalize(dh, dh);
            v3_scale(F_des, dh, Tmax * sinf(DIVERT_TILT));
            F_des[2] = Tmax * cosf(DIVERT_TILT);
        } else {
            (void)thrust_project(F_des, Tmin, Tmax, BURN_TILT_MAX, F_des);
        }
        (void)v3_normalize(u, F_des);
        thrust_attitude(u, S.heading_flip, q_des, d_b);
        q_rotate(d_L, S.nav.q, d_b);
        q_error(th, S.nav.q, q_des);
        if (S.mode == MODE_FLIP) {
            /* pitch up on flaps first (the aft-heavy body wants to go tail-first), then light
             * the engines once the thrust axis is mostly up or the pre-flip time is spent;
             * hold minimum thrust until the attitude converges */
            if (count_active() == 0 && (S.t_mode >= FLIP_PRE_TIME || v3_norm(th) < FLIP_IGNITE_ERR)) {
                engine_manage(0.0f, 2, dt);     /* light the burn set only: smaller flip impulse */
            }
            T_cmd = Tmin;
            gains = &GAINS_FLIP;
            flap_trim = 0.5f;
            if (v3_norm(th) < FLIP_DONE_ERR) { enter_mode(MODE_BURN); }
            else if (S.t_mode > FLIP_WATCHDOG_T ||
                     S.nav.p[2] - S.target_cg[2] < FLIP_WATCHDOG_H) {
                /* flip overrunning: commit to the burn with whatever attitude we have */
                engine_manage(0.0f, 2, dt);
                enter_mode(MODE_BURN);
            }
        } else {
            T_cmd = S.divert_priority ? Tmax : gnc_clampf(v3_dot(F_des, d_L), Tmin, Tmax);
            gains = S.divert_priority ? &GAINS_FLIP : &GAINS_BURN;
            flap_trim = 0.2f;
            use_flaps = 0;      /* tail-first flow: flaps have no authority, avoid chatter */
            if (!S.divert_priority && (S.pdg.tgo <= TGO_HANDOVER || S.nav.p[2] - S.target_cg[2] <= 0.5f * S.cfg.h_gate)) {
                enter_mode(MODE_TERMINAL);
            }
        }
        break;
    }
    case MODE_TERMINAL: {
        float a_drift[3], Tmin, Tmax, u[3], d_b[3], d_L[3];
        v3_add(a_drift, g_vec, a_aero_L);
        terminal_accel(S.nav.p, S.nav.v, S.target_cg, S.cfg.h_gate, S.cfg.v_gate, a_drift, a_cmd);
        v3_scale(F_des, a_cmd, S.mass);
        if (newly_failed && count_healthy() == 0) { enter_mode(MODE_ABORT_NO_THRUST); break; }
        /* stay on the burn engines until the demand fits on one engine, then shut down */
        engine_manage(v3_norm(F_des),
                      (v3_norm(F_des) < ENG_SINGLE_FRAC * VEH_T_MAX_EACH || S.t_mode > 4.0f) ? 1 : 2, dt);
        thrust_bounds(&Tmin, &Tmax);
        {
            const float hagl = S.nav.p[2] - S.target_cg[2];
            const float tilt = TERM_TILT_MIN + (TERM_TILT_MAX - TERM_TILT_MIN)
                               * gnc_clampf(hagl / TERM_TILT_TAPER_H, 0.0f, 1.0f);
            (void)thrust_project(F_des, Tmin, Tmax, tilt, F_des);
        }
        (void)v3_normalize(u, F_des);
        thrust_attitude(u, S.heading_flip, q_des, d_b);
        q_rotate(d_L, S.nav.q, d_b);
        T_cmd = gnc_clampf(v3_dot(F_des, d_L), Tmin, Tmax);
        gains = &GAINS_BURN;
        flap_trim = 0.2f;
        use_flaps = 0;
        if (in->touchdown || (S.nav.p[2] - S.target_cg[2] < 0.3f && fabsf(S.nav.v[2]) < 0.5f)) {
            enter_mode(MODE_LANDED);
        }
        break;
    }
    case MODE_LANDED:
    case MODE_ABORT_NO_THRUST:
    default:
        for (i = 0; i < FSW_N_ENG; ++i) { S.eng_on[i] = 0; }
        use_flaps = 0;
        gains = &GAINS_BURN;
        break;
    }
    if (S.mode == MODE_LANDED) {
        for (i = 0; i < FSW_N_ENG; ++i) { S.eng_on[i] = 0; }
    }

    /* ================= attitude control & allocation ================= */
    att_ctrl_step(&S.att, gains, q_des, S.nav.q, S.nav.omega_b, dt, tau_req);
    v3_sub(tau_act, tau_req, M_body);                       /* cancel predicted body aero moment */
    alloc_torque(tau_act, T_total, tau_off, vair_b, rho, flap_trim, use_flaps, &al);
    S.att.hold_integrator = al.saturated;
    if (S.mode == MODE_LANDED || S.mode == MODE_ABORT_NO_THRUST) {
        al.gimbal[0] = 0.0f; al.gimbal[1] = 0.0f; v3_zero(al.rcs);
    }

    /* ================= throttle ================= */
    {
        const int32_t n_act = count_active();
        S.throttle_cmd = (n_act > 0) ? gnc_clampf(T_cmd / ((float)n_act * VEH_T_MAX_EACH), VEH_THROTTLE_MIN, 1.0f)
                                     : VEH_THROTTLE_MIN;
    }

    /* ================= outputs ================= */
    for (i = 0; i < FSW_N_ENG; ++i) {
        cmd->engine_enable[i] = (S.eng_on[i] && !S.eng_failed[i]) ? 1 : 0;
        cmd->throttle[i] = S.throttle_cmd;
    }
    cmd->gimbal[0] = al.gimbal[0];
    cmd->gimbal[1] = al.gimbal[1];
    for (i = 0; i < FSW_N_FLAP; ++i) { cmd->flaps[i] = al.flaps[i]; S.flaps_cmd[i] = al.flaps[i]; }
    v3_copy(cmd->rcs, al.rcs);
    S.gimbal_cmd[0] = al.gimbal[0];
    S.gimbal_cmd[1] = al.gimbal[1];


    /* ================= telemetry ================= */
    if (tlm != 0) {
        tlm->t = S.t;
        tlm->mode = S.mode;
        tlm->target_id = S.target_id;
        v3_copy(tlm->nav_p, S.nav.p);
        v3_copy(tlm->nav_v, S.nav.v);
        q_copy(tlm->nav_q, S.nav.q);
        v3_copy(tlm->nav_bg, S.nav.bg);
        v3_copy(tlm->nav_ba, S.nav.ba);
        for (i = 0; i < NAV_NX; ++i) { tlm->nav_sig[i] = sqrtf(fmaxf(S.nav.P[i * NAV_NX + i], 0.0f)); }
        for (i = 0; i < NAV_N_SENS; ++i) {
            tlm->nis[i] = S.nav.health[i].last_nis;
            tlm->sens_failed[i] = S.nav.health[i].failed;
        }
        tlm->wind_est[0] = S.wind.x[0];
        tlm->wind_est[1] = S.wind.x[1];
        tlm->wind_sig[0] = sqrtf(S.wind.P[0]);
        tlm->wind_sig[1] = sqrtf(S.wind.P[4]);
        tlm->wind_k = S.wind.x[2];
        tlm->mass_est = S.mass;
        q_copy(tlm->q_des, q_des);
        v3_copy(tlm->att_err, S.att.th_err);
        v3_copy(tlm->F_des, F_des);
        v3_copy(tlm->a_cmd, a_cmd);
        v3_copy(tlm->a_aero, a_aero_L);
        tlm->tgo = S.pdg.tgo;
        tlm->h_ign = S.h_ign;
        tlm->mu_abort = S.mu_abort;
        v3_copy(tlm->target, S.target);
        tlm->n_eng_healthy = count_healthy();
        tlm->n_eng_active = count_active();
        for (i = 0; i < FSW_N_ENG; ++i) { tlm->eng_failed[i] = S.eng_failed[i]; }
        tlm->alloc_saturated = al.saturated;
        tlm->T_cmd = T_cmd;
        v3_copy(tlm->tau_req, tau_req);
        tlm->divert_priority = S.divert_priority;
    }
}
