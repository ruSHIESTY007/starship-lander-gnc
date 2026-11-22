/**
 * @file fsw_config.h
 * @brief Compile-time vehicle model (as known to the flight software) and tuning constants.
 *
 * These are the *nominal* values the FSW believes. The truth simulation disperses the real
 * vehicle around them (sim/params.py); tests/test_sil.py cross-checks that the onboard
 * aero/atmosphere models reproduce the truth models exactly at nominal parameters.
 */
#ifndef FSW_CONFIG_H
#define FSW_CONFIG_H

#define FSW_N_ENG   3
#define FSW_N_FLAP  4
#define FSW_N_STRIP 5
#define FSW_DT      0.01f          /* 100 Hz executive */

/* ---------------- environment ---------------- */
#define FSW_G0      9.80665f

/* ---------------- vehicle (nominal) ---------------- */
#define VEH_M_DRY           100000.0f
#define VEH_M_WET           130000.0f
#define VEH_IXX             1.6e6f
#define VEH_IYY             2.6e7f
#define VEH_IZZ             2.6e7f
#define VEH_X_AFT           (-23.0f)
#define VEH_DIAMETER        9.0f
#define VEH_T_MAX_EACH      1.6e6f
#define VEH_THROTTLE_MIN    0.40f
#define VEH_ISP             330.0f
#define VEH_ENG_RING_R      1.6f
#define VEH_GIMBAL_MAX      0.258819f  /* sin(15 deg) as lateral thrust fraction */
#define VEH_SPOOL_UP_TIME   1.5f
#define VEH_STRIP_LEN       10.0f
#define VEH_CA_NOSE         0.30f
#define VEH_CA_TAIL         0.85f
#define VEH_FLAP_CF         1.28f
#define VEH_FLAP_DIHEDRAL   0.174533f  /* 10 deg */
#define VEH_RCS_MAX         4.0e5f
#define VEH_LEG_X           (-24.0f)   /* leg foot plane along x_B relative to CoM */

/* strip stations and flap geometry: see fsw_config.c */
extern const float VEH_STRIP_X[FSW_N_STRIP];
extern const float VEH_FLAP_X[FSW_N_FLAP];
extern const float VEH_FLAP_Y[FSW_N_FLAP];
extern const float VEH_FLAP_AREA[FSW_N_FLAP];
extern const float VEH_ENG_ANGLE[FSW_N_ENG];

/* ---------------- navigation tuning ---------------- */
#define NAV_GYRO_ARW        4.4e-5f    /* rad/sqrt(s)        */
#define NAV_ACCEL_VRW       5.0e-4f    /* m/s/sqrt(s)        */
#define NAV_GYRO_BIAS_RW    1.0e-7f
#define NAV_ACCEL_BIAS_RW   1.0e-5f
#define NAV_VEL_PROC_EXTRA  0.02f      /* unmodelled vibration / lever arm [m/s/sqrt(s)] */
#define NAV_POS_PROC        0.01f
#define NAV_GNSS_POS_SIG_H  1.5f       /* inflated for correlated GNSS bias (not estimated) */
#define NAV_GNSS_POS_SIG_V  2.6f
#define NAV_GNSS_VEL_SIG    0.08f
#define NAV_RADAR_SIG_ABS   0.35f
#define NAV_RADAR_SIG_REL   0.006f
#define NAV_RADAR_MAX_OFFNADIR_COS 0.819152f   /* cos 35 deg */
#define NAV_CHI2_1DOF       10.83f     /* 99.9 % */
#define NAV_CHI2_3DOF       16.27f     /* 99.9 % */
#define NAV_FAIL_CONSEC     20         /* consecutive rejections before a sensor is latched failed */

/* ---------------- wind estimator tuning ---------------- */
#define WIND_Q_W            0.5f       /* wind random walk  [m/s/sqrt(s)] */
#define WIND_Q_K            0.003f     /* drag-scale random walk [1/sqrt(s)] */
#define WIND_AERO_MODEL_ERR 0.12f      /* relative aero-model error, 1 sigma */
#define WIND_THRUST_ERR_AX  0.02f      /* relative thrust error along x_B */
#define WIND_THRUST_ERR_LAT 0.004f     /* lateral thrust error fraction (misalignment) */
#define WIND_QBAR_MIN       300.0f     /* Pa, below which aero is too weak to invert */

#endif /* FSW_CONFIG_H */
