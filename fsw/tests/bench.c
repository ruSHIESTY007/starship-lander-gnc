/**
 * @file bench.c
 * @brief Host timing of the flight software: full 100 Hz step per mode and key kernels.
 *        (A Cortex-M7 at 400-600 MHz is typically 5-15x slower than a desktop core.)
 */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "control.h"
#include "fsw.h"
#include "gnc_math.h"
#include "guidance.h"
#include "nav.h"
#include "nn_policy.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static volatile float g_sink;

int main(void)
{
    fsw_init_t ini;
    fsw_input_t in;
    fsw_cmd_t cmd;
    fsw_tlm_t tlm;
    const int N = 20000;
    double t0, worst = 0.0, total = 0.0;
    int k;
    memset(&ini, 0, sizeof ini);
    memset(&in, 0, sizeof in);
    v3_set(ini.p0, -3000.0f, 300.0f, 9000.0f);
    v3_set(ini.v0, 70.0f, 0.0f, -100.0f);
    q_from_euler321(ini.q0, 0.0f, 0.0f, GNC_PI);
    ini.sig_pos = 10.0f; ini.sig_vel = 0.5f; ini.sig_att = 0.01f;
    ini.mass0 = 130000.0f; ini.gamma = 6.0f; ini.h_gate = 40.0f; ini.v_gate = 3.0f;
    ini.use_wind_ff = 1;
    fsw_init(&ini);
    in.accel[2] = -9.8f;          /* belly-flop: drag along -z_B */
    for (k = 0; k < 3; ++k) { in.eng_ready[k] = 1; }
    for (k = 0; k < N; ++k) {
        double dt;
        in.t = 0.01f * (float)k;
        in.gnss_valid = (k % 10 == 0);
        v3_set(in.gnss_pos, -3000.0f + 0.7f * (float)k * 0.01f, 300.0f, 9000.0f - (float)k * 0.01f * 80.0f);
        v3_set(in.gnss_vel, 70.0f, 0.0f, -100.0f);
        in.radar_valid = (k % 5 == 0);
        in.radar_range = 2000.0f;
        t0 = now_us();
        fsw_step(&in, &cmd, &tlm);
        dt = now_us() - t0;
        total += dt;
        if (k > 100 && dt > worst) { worst = dt; }
    }
    printf("fsw_step (glide, GNSS 10 Hz, radar 20 Hz, wind EKF 20 Hz): mean %.2f us, worst %.2f us\n",
           total / N, worst);

    /* kernels */
    {
        nav_t nav;
        nav_init_sigma_t sig = { 1, 0.1f, 0.01f, 1e-5f, 1e-2f };
        float p[3] = { 0, 0, 500 }, v[3] = { 0, 0, -50 }, q[4] = { 1, 0, 0, 0 }, gy[3] = { 0.01f, 0, 0 }, ac[3] = { 0, 0, 9.8f };
        nav_init(&nav, p, v, q, &sig);
        t0 = now_us();
        for (k = 0; k < N; ++k) { nav_propagate(&nav, gy, ac, 0.01f); }
        printf("nav_propagate (18-state ESKF covariance):   %.2f us\n", (now_us() - t0) / N);
        {
            static nav_t tmpl, work;
            double t_copy, t_both;
            tmpl = nav;
            t0 = now_us();
            for (k = 0; k < N; ++k) { work = tmpl; g_sink = work.p[0]; }
            t_copy = now_us() - t0;
            t0 = now_us();
            for (k = 0; k < N; ++k) { work = tmpl; (void)nav_update_gnss_pos(&work, p); }
            t_both = now_us() - t0;
            printf("nav_update_gnss_pos (gate + 3 scalar upd):  %.2f us\n", (t_both - t_copy) / N);
        }
    }
    {
        pdg_t g;
        const float rf[3] = { 0, 0, 64 }, vf[3] = { 0, 0, -3 }, a[3] = { 0, 0, -9.0f };
        float r[3] = { -30, 20, 330 }, v[3] = { 2, -1, -42 };
        pdg_init(&g, 6.0f, rf, vf);
        t0 = now_us();
        for (k = 0; k < 2000; ++k) { g.initialized = 0; pdg_step(&g, r, v, a, 0.01f); }
        printf("pdg_step, full tgo search (32 grid + 24 GS): %.2f us\n", (now_us() - t0) / 2000);
        t0 = now_us();
        g.initialized = 1; g.tgo = 8.0f;
        for (k = 0; k < N; ++k) { g.tgo = 8.0f; pdg_step(&g, r, v, a, 0.01f); }
        printf("pdg_step, bounded re-optimisation:          %.2f us\n", (now_us() - t0) / N);
    }
    {
        static float params[NN_N_PARAMS];
        float x[NN_IN] = { 1, 2, 300, 0, 0, -40, 1, 1.28f }, y[NN_OUT];
        for (k = 0; k < NN_N_PARAMS; ++k) { params[k] = 0.01f * (float)((k * 7919) % 200 - 100) / 100.0f; }
        for (k = 0; k < NN_IN; ++k) { params[NN_N_PARAMS - 2 * NN_OUT - NN_IN + k] = 1.0f; }
        (void)nn_policy_load(params, NN_N_PARAMS);
        t0 = now_us();
        for (k = 0; k < N; ++k) { nn_policy_eval(x, y); g_sink = y[0]; }
        printf("nn_policy_eval (8-64-64-3 MLP, %d params):  %.2f us\n", NN_N_PARAMS, (now_us() - t0) / N);
    }
    return 0;
}
