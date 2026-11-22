"""
Vehicle, environment and scenario parameters for the truth simulation.

All values are notional round numbers chosen to give a physically plausible belly-flop
vehicle. Frames:

* L (local-level, landing frame): origin at primary pad, x East (nominal approach direction),
  y North, z Up. Flat, non-rotating Earth (justified for a ~10 km, ~2 min terminal phase;
  Coriolis acceleration ~1e-2 m/s^2 is absorbed by closed-loop guidance).
* B (body): origin at CoM, +x toward the nose along the long axis, +z out of the belly,
  +y completes the right-handed set. Engines push along +x_B. Belly-flop flight has the
  belly (+z_B) facing the ground, so the relative wind arrives along +z_B (alpha ~ 90 deg).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G0 = 9.80665               # standard gravity [m/s^2] (also used for Isp)
G_EARTH = 9.80665          # local gravity magnitude (constant over a 10 km descent; <0.4% change)


@dataclass
class VehicleParams:
    # ---------------- geometry / mass ----------------
    length: float = 50.0
    diameter: float = 9.0
    x_aft: float = -23.0              # aft end (engine plane) relative to CoM [m]
    x_nose: float = 27.0
    m_dry: float = 100_000.0
    m_prop: float = 30_000.0          # landing propellant at glide start
    inertia: tuple = (1.6e6, 2.6e7, 2.6e7)   # principal, about CoM [kg m^2]

    # ---------------- engines ----------------
    n_engines: int = 3
    T_max_each: float = 1.6e6         # sea-level max thrust per engine [N]
    throttle_min: float = 0.40
    isp: float = 330.0                # sea-level Isp [s]
    engine_ring_radius: float = 1.6   # engines on a ring in the B y-z plane
    engine_ring_angles_deg: tuple = (90.0, 210.0, 330.0)
    gimbal_max_deg: float = 15.0
    gimbal_rate_deg_s: float = 30.0
    gimbal_tau: float = 0.04
    throttle_tau: float = 0.15
    spool_up_time: float = 1.5        # ignition to commanded thrust (linear ramp) [s]
    shutdown_time: float = 0.3

    # ---------------- aerodynamics ----------------
    # Crossflow body strips (centers along x_B, each `strip_len` long, `diameter` wide)
    strip_x: tuple = (-18.0, -8.0, 2.0, 12.0, 22.0)
    strip_len: float = 10.0
    ca_nose_first: float = 0.30       # axial force coeff., flow onto nose (ref: base area)
    ca_tail_first: float = 0.85       # axial force coeff., flow onto engine end
    cdc_scale: float = 1.0            # multiplies crossflow drag table (dispersed in MC)
    # Flaps: [fore-right, fore-left, aft-right, aft-left]
    flap_x: tuple = (19.0, 19.0, -17.0, -17.0)
    flap_y: tuple = (7.0, -7.0, 7.5, -7.5)
    flap_area: tuple = (30.0, 30.0, 45.0, 45.0)
    flap_dihedral_deg: float = 10.0
    flap_cf: float = 1.28
    flap_tau: float = 0.10
    flap_rate: float = 1.5            # exposure fraction per second

    # ---------------- RCS ----------------
    rcs_torque_max: tuple = (4.0e5, 4.0e5, 4.0e5)

    # ---------------- truth-only error sources (dispersed in Monte Carlo) ----------------
    thrust_scale: tuple = (1.0, 1.0, 1.0)          # per-engine actual/nominal thrust
    isp_scale: float = 1.0
    thrust_misalign: tuple = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))  # lateral fractions
    com_offset: tuple = (0.0, 0.0, 0.0)            # true CoM minus design CoM, body [m]

    # ---------------- landing gear ----------------
    leg_x: float = -24.0
    leg_radius: float = 5.5
    n_legs: int = 6

    @property
    def m_wet(self) -> float:
        return self.m_dry + self.m_prop

    @property
    def strip_area(self) -> float:
        return self.strip_len * self.diameter

    @property
    def base_area(self) -> float:
        return np.pi * (0.5 * self.diameter) ** 2

    def engine_positions(self) -> np.ndarray:
        a = np.deg2rad(np.asarray(self.engine_ring_angles_deg))
        rr = self.engine_ring_radius
        return np.stack([np.full_like(a, self.x_aft), rr * np.cos(a), rr * np.sin(a)], axis=1)

    def flap_positions(self) -> np.ndarray:
        return np.stack([self.flap_x, self.flap_y, np.zeros(4)], axis=1)

    def flap_normals(self) -> np.ndarray:
        g = np.deg2rad(self.flap_dihedral_deg)
        side = np.sign(np.asarray(self.flap_y))
        return np.stack([np.zeros(4), side * np.sin(g), np.full(4, np.cos(g))], axis=1)

    def leg_positions(self) -> np.ndarray:
        a = np.linspace(0, 2 * np.pi, self.n_legs, endpoint=False)
        return np.stack([np.full_like(a, self.leg_x),
                         self.leg_radius * np.cos(a), self.leg_radius * np.sin(a)], axis=1)


@dataclass
class SensorParams:
    imu_rate: float = 100.0
    gyro_arw: float = 4.4e-5          # rad/sqrt(s)   (0.15 deg/sqrt(h))
    gyro_bias_sigma: float = 4.8e-6   # rad/s         (1 deg/h turn-on bias)
    gyro_bias_rw: float = 1.0e-7      # rad/s/sqrt(s)
    accel_vrw: float = 5.0e-4         # m/s/sqrt(s)
    accel_bias_sigma: float = 0.01    # m/s^2 (~1 mg)
    accel_bias_rw: float = 1.0e-5     # m/s^2/sqrt(s)
    gnss_rate: float = 10.0
    gnss_pos_sigma: tuple = (1.0, 1.0, 2.0)
    gnss_vel_sigma: float = 0.05
    gnss_bias_sigma: float = 0.5      # Gauss-Markov position bias
    gnss_bias_tau: float = 100.0
    radar_rate: float = 20.0
    radar_max_range: float = 2500.0
    radar_sigma_abs: float = 0.3
    radar_sigma_rel: float = 0.005
    radar_max_offnadir_deg: float = 35.0
    pc_sigma: float = 0.01            # chamber pressure telemetry, fraction of nominal


@dataclass
class EnvParams:
    rho_scale: float = 1.0            # density dispersion multiplier
    temp_offset: float = 0.0          # K, shifts speed of sound
    wind_surface: tuple = (6.0, 2.0)  # mean wind at 10 m AGL, L-frame (x, y) [m/s]
    wind_aloft: tuple = (15.0, -4.0)  # mean wind at 10 km
    turb_sigma: float = 1.5           # turbulence intensity [m/s]
    gust: tuple | None = None         # (t_start, duration, (gx, gy, gz)) 1-cosine gust


@dataclass
class FaultParams:
    engine_fail_time: float | None = None
    engine_fail_index: int = 0
    gnss_dropout: tuple | None = None     # (t0, t1)
    gnss_step_bias: tuple | None = None   # (t0, (bx, by, bz)) persistent step
    radar_stuck_time: float | None = None
    pc_stuck_time: float | None = None    # Pc of failed engine keeps reading nominal


@dataclass
class Scenario:
    r0: tuple = (-3200.0, 350.0, 10_000.0)
    v0: tuple = (75.0, -4.0, -115.0)
    yaw0_deg: float = 0.0             # nose heading at glide start (toward +x_L)
    pitch0_deg: float = 0.0
    roll0_deg: float = 0.0
    pad: tuple = (0.0, 0.0, 0.0)
    backup: tuple = (450.0, 0.0, 0.0)
    pad_radius: float = 10.0
    t_max: float = 260.0
    dt_fsw: float = 0.01
    substeps: int = 2                 # RK4 substeps per FSW frame (200 Hz truth)
    seed: int = 0
    veh: VehicleParams = field(default_factory=VehicleParams)
    sens: SensorParams = field(default_factory=SensorParams)
    env: EnvParams = field(default_factory=EnvParams)
    faults: FaultParams = field(default_factory=FaultParams)


# Touchdown acceptance criteria
TD_MAX_VZ = 2.5        # m/s
TD_MAX_VH = 1.5        # m/s
TD_MAX_TILT_DEG = 5.0
TD_MAX_RATE_DEG_S = 5.0
