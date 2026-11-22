"""
Sensor models: strapdown IMU, GNSS (pos/vel), dual-antenna radar altimeter, engine
chamber-pressure telemetry, and touchdown (leg) switches -- with fault injection hooks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import quat
from .params import FaultParams, SensorParams


@dataclass
class SensorFrame:
    t: float = 0.0
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gnss_valid: bool = False
    gnss_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gnss_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radar_valid: bool = False
    radar_antenna: int = 0          # 0 = belly (+z_B boresight), 1 = aft (-x_B boresight)
    radar_range: float = 0.0
    pc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eng_ready: np.ndarray = field(default_factory=lambda: np.ones(3, bool))
    touchdown: bool = False


class SensorSuite:
    def __init__(self, sp: SensorParams, faults: FaultParams, rng: np.random.Generator, dt: float):
        self.sp, self.faults, self.rng, self.dt = sp, faults, rng, dt
        self.bg = rng.normal(0.0, sp.gyro_bias_sigma, 3)
        self.ba = rng.normal(0.0, sp.accel_bias_sigma, 3)
        self.gnss_bias = rng.normal(0.0, sp.gnss_bias_sigma, 3)
        self.k = 0
        self.radar_last = None
        self.pc_last = None

    def sample(self, t, x, f_b, pc_true, legs_contact: bool, eng_failed=None) -> SensorFrame:
        sp, rng, dt = self.sp, self.rng, self.dt
        r, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
        fr = SensorFrame(t=t)
        # ---- IMU (every frame) ----
        self.bg += rng.normal(0.0, sp.gyro_bias_rw * math.sqrt(dt), 3)
        self.ba += rng.normal(0.0, sp.accel_bias_rw * math.sqrt(dt), 3)
        fr.gyro = w + self.bg + rng.normal(0.0, sp.gyro_arw / math.sqrt(dt), 3)
        fr.accel = f_b + self.ba + rng.normal(0.0, sp.accel_vrw / math.sqrt(dt), 3)
        # ---- GNSS ----
        n_gnss = int(round(1.0 / (sp.gnss_rate * dt)))
        phi = math.exp(-dt / sp.gnss_bias_tau)
        self.gnss_bias = phi * self.gnss_bias + sp.gnss_bias_sigma * math.sqrt(1 - phi * phi) * rng.standard_normal(3)
        if self.k % n_gnss == 0:
            drop = self.faults.gnss_dropout
            if not (drop and drop[0] <= t <= drop[1]):
                fr.gnss_valid = True
                fr.gnss_pos = r + self.gnss_bias + rng.normal(0.0, sp.gnss_pos_sigma)
                fr.gnss_vel = v + rng.normal(0.0, sp.gnss_vel_sigma, 3)
                step = self.faults.gnss_step_bias
                if step and t >= step[0]:
                    fr.gnss_pos = fr.gnss_pos + np.asarray(step[1], float)
        # ---- radar altimeter ----
        n_rad = int(round(1.0 / (sp.radar_rate * dt)))
        if self.k % n_rad == 0:
            R = quat.to_dcm(q)
            bores = (R[:, 2], -R[:, 0])           # belly, aft boresights in L
            cosines = [-b[2] for b in bores]       # cos of angle from nadir
            ant = int(np.argmax(cosines))
            c = cosines[ant]
            h = r[2]
            if c > math.cos(math.radians(sp.radar_max_offnadir_deg)) and 0.0 < h / c < sp.radar_max_range:
                rng_true = h / c
                meas = rng_true + rng.normal(0.0, sp.radar_sigma_abs + sp.radar_sigma_rel * rng_true)
                if self.faults.radar_stuck_time is not None and t >= self.faults.radar_stuck_time:
                    if self.radar_last is None:
                        self.radar_last = (ant, meas)
                    ant, meas = self.radar_last
                fr.radar_valid, fr.radar_antenna, fr.radar_range = True, ant, meas
        # ---- chamber pressure ----
        pc = pc_true + rng.normal(0.0, sp.pc_sigma, pc_true.shape)
        if self.faults.pc_stuck_time is not None and t >= self.faults.pc_stuck_time:
            if self.pc_last is None:
                self.pc_last = pc.copy()
            i = self.faults.engine_fail_index
            pc[i] = self.pc_last[i]
        fr.pc = pc
        if eng_failed is not None:
            fr.eng_ready = ~np.asarray(eng_failed, bool)
        fr.touchdown = legs_contact
        self.k += 1
        return fr
