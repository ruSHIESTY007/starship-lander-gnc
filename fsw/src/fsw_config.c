/** @file fsw_config.c  @brief Geometry tables of the nominal vehicle model. */
#include "fsw_config.h"

const float VEH_STRIP_X[FSW_N_STRIP]  = { -18.0f, -8.0f, 2.0f, 12.0f, 22.0f };
const float VEH_FLAP_X[FSW_N_FLAP]    = { 19.0f, 19.0f, -17.0f, -17.0f };   /* FR, FL, AR, AL */
const float VEH_FLAP_Y[FSW_N_FLAP]    = { 7.0f, -7.0f, 7.5f, -7.5f };
const float VEH_FLAP_AREA[FSW_N_FLAP] = { 30.0f, 30.0f, 45.0f, 45.0f };
const float VEH_ENG_ANGLE[FSW_N_ENG]  = { 1.5707963f, 3.6651914f, 5.7595865f }; /* 90,210,330 deg */
