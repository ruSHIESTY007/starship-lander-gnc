"""
Actuator models: engines (start/stop transients, throttle lag, failures, common 2-axis
gimbal with lag + rate limit), flaps (lag + rate limit) and RCS torque (lag + saturation).
"""
from __future__ import annotations

import math

import numpy as np

from .params import G0, VehicleParams

OFF, STARTING, RUNNING, SHUTDOWN, FAILED = range(5)


def _rate_lag(x, cmd, tau, rate, dt):
    """First-order lag followed by a rate limit (elementwise)."""
    target = x + (cmd - x) * (1.0 - math.exp(-dt / tau))
    return x + np.clip(target - x, -rate * dt, rate * dt)


class Engines:
    def __init__(self, veh: VehicleParams):
        self.veh = veh
        n = veh.n_engines
        self.T = np.zeros(n)
        self.mode = np.full(n, OFF)
        self.fail_flag = np.zeros(n, bool)
        self.gimbal = np.zeros(2)          # lateral thrust fractions (y_B, z_B)
        self.r = veh.engine_positions() - np.asarray(veh.com_offset, float)
        self.Tmax = veh.T_max_each * np.asarray(veh.thrust_scale, float)
        self.mis = np.asarray(veh.thrust_misalign, float)
        self.g_max = math.sin(math.radians(veh.gimbal_max_deg))
        self.g_rate = math.radians(veh.gimbal_rate_deg_s)
        self.n_starts = np.zeros(n, int)

    def fail(self, i: int):
        self.fail_flag[i] = True
        self.mode[i] = FAILED

    def step(self, dt, enable, throttle, gimbal_cmd, prop_left: float):
        veh = self.veh
        thr = np.clip(np.asarray(throttle, float), veh.throttle_min, 1.0)
        for i in range(veh.n_engines):
            target = thr[i] * self.Tmax[i] if prop_left > 0.0 else 0.0
            m = self.mode[i]
            if m == FAILED:
                self.T[i] *= math.exp(-dt / 0.1)
                continue
            if enable[i] and prop_left > 0.0:
                if m in (OFF, SHUTDOWN):
                    if m == OFF:
                        self.n_starts[i] += 1
                    self.mode[i] = m = STARTING
                if m == STARTING:
                    self.T[i] = min(self.T[i] + self.Tmax[i] * dt / veh.spool_up_time, target)
                    if self.T[i] >= target - 1e-6:
                        self.mode[i] = RUNNING
                else:
                    self.T[i] += (target - self.T[i]) * (1.0 - math.exp(-dt / veh.throttle_tau))
            else:
                if m != OFF:
                    self.mode[i] = SHUTDOWN
                    self.T[i] = max(self.T[i] - self.Tmax[i] * dt / veh.shutdown_time, 0.0)
                    if self.T[i] <= 0.0:
                        self.mode[i] = OFF
        g = np.asarray(gimbal_cmd, float)
        gn = np.linalg.norm(g)
        if gn > self.g_max:
            g = g * (self.g_max / gn)
        self.gimbal = _rate_lag(self.gimbal, g, veh.gimbal_tau, self.g_rate, dt)

    def forces(self):
        F = np.zeros(3)
        M = np.zeros(3)
        for i in range(self.veh.n_engines):
            if self.T[i] <= 0.0:
                continue
            gy = self.gimbal[0] + self.mis[i, 0]
            gz = self.gimbal[1] + self.mis[i, 1]
            d = np.array([math.sqrt(max(1.0 - gy * gy - gz * gz, 0.0)), gy, gz])
            f = self.T[i] * d
            F += f
            ri = self.r[i]
            M += (ri[1] * f[2] - ri[2] * f[1], ri[2] * f[0] - ri[0] * f[2], ri[0] * f[1] - ri[1] * f[0])
        return F, M

    def mdot(self):
        return -self.T.sum() / (self.veh.isp * self.veh.isp_scale * G0)

    def chamber_pressure(self):
        """Normalised chamber pressure ~ thrust / nominal max thrust."""
        return self.T / self.veh.T_max_each


class Flaps:
    def __init__(self, veh: VehicleParams):
        self.veh = veh
        self.e = np.full(4, 0.5)

    def step(self, dt, cmd):
        cmd = np.clip(np.asarray(cmd, float), 0.0, 1.0)
        self.e = np.clip(_rate_lag(self.e, cmd, self.veh.flap_tau, self.veh.flap_rate, dt), 0.0, 1.0)


class RCS:
    def __init__(self, veh: VehicleParams):
        self.lim = np.asarray(veh.rcs_torque_max, float)
        self.tau = np.zeros(3)

    def step(self, dt, cmd):
        cmd = np.clip(np.asarray(cmd, float), -self.lim, self.lim)
        self.tau += (cmd - self.tau) * (1.0 - math.exp(-dt / 0.02))
