"""
Closed-loop simulation executive.

Frame loop at the flight-software rate (default 100 Hz):
  1. sample sensors at the current truth state
  2. call the controller (the C flight software via ctypes, or a Python test controller)
  3. hold the actuator command and integrate truth over `substeps` RK4 steps, updating
     actuator dynamics, turbulence and fault injection at every substep
  4. detect ground contact and classify the landing
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import quat
from .actuators import RCS, Engines, Flaps
from .atmosphere import us76
from .dynamics import RigidBody
from .params import (TD_MAX_RATE_DEG_S, TD_MAX_TILT_DEG, TD_MAX_VH, TD_MAX_VZ, Scenario)
from .sensors import SensorFrame, SensorSuite
from .wind import WindModel


@dataclass
class ActuatorCmd:
    engine_enable: np.ndarray = field(default_factory=lambda: np.zeros(3, bool))
    throttle: np.ndarray = field(default_factory=lambda: np.full(3, 0.4))
    gimbal: np.ndarray = field(default_factory=lambda: np.zeros(2))
    flaps: np.ndarray = field(default_factory=lambda: np.full(4, 0.5))
    rcs: np.ndarray = field(default_factory=lambda: np.zeros(3))


class Controller(Protocol):
    def step(self, frame: SensorFrame) -> tuple[ActuatorCmd, dict]: ...


def initial_state(sc: Scenario) -> np.ndarray:
    x = np.zeros(14)
    x[0:3] = sc.r0
    x[3:6] = sc.v0
    # roll0 is measured from the belly-down attitude
    x[6:10] = quat.from_euler321(math.radians(sc.yaw0_deg), math.radians(sc.pitch0_deg),
                                 math.pi + math.radians(sc.roll0_deg))
    x[13] = sc.veh.m_wet
    return x


class Simulation:
    def __init__(self, sc: Scenario, controller: Controller, log_every: int = 1):
        self.sc = sc
        self.ctrl = controller
        self.rng = np.random.default_rng(sc.seed)
        self.body = RigidBody(sc.veh, sc.env)
        self.eng = Engines(sc.veh)
        self.flaps = Flaps(sc.veh)
        self.rcs = RCS(sc.veh)
        self.wind = WindModel(sc.env, self.rng)
        self.sens = SensorSuite(sc.sens, sc.faults, self.rng, sc.dt_fsw)
        self.x = initial_state(sc)
        self.log_every = log_every
        veh = sc.veh
        off = np.asarray(veh.com_offset, float)
        a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        rr = 0.5 * veh.diameter
        hull = [np.array([veh.x_nose, 0.0, 0.0])]
        hull += [np.array([veh.x_aft, rr * math.cos(t), rr * math.sin(t)]) for t in a]
        hull += [np.array([veh.x_nose - 5.0, rr * math.cos(t), rr * math.sin(t)]) for t in a]
        self.legs_b = veh.leg_positions() - off
        self.hull_b = np.array(hull) - off

    # ------------------------------------------------------------------
    def _contact(self, x):
        R = quat.to_dcm(x[6:10])
        leg_z = x[2] + (self.legs_b @ R.T)[:, 2]
        hull_z = x[2] + (self.hull_b @ R.T)[:, 2]
        return leg_z.min() <= 0.0, hull_z.min() <= 0.0

    def run(self) -> dict:
        sc = self.sc
        dt = sc.dt_fsw
        h = dt / sc.substeps
        n_max = int(sc.t_max / dt)
        flt = sc.faults
        logs: dict[str, list] = {k: [] for k in (
            "t", "r", "v", "q", "w", "m", "T", "gimbal", "flaps", "wind", "qbar", "mach", "alpha")}
        tlm_log: dict[str, list] = {}
        outcome = "timeout"
        td = None
        f_b = np.zeros(3)
        x = self.x
        cmd = ActuatorCmd()
        info = (0.0, 0.0, 0.0)
        for k in range(n_max):
            t = k * dt
            if flt.engine_fail_time is not None and t >= flt.engine_fail_time \
                    and not self.eng.fail_flag[flt.engine_fail_index]:
                self.eng.fail(flt.engine_fail_index)
            # specific force at the sample instant (consistent with current actuators)
            F_e, M_e = self.eng.forces()
            wind_L = self.wind.total(t, x[2])
            _, f_b, info = self.body.deriv(x, F_e, M_e + self.rcs.tau, self.eng.mdot(), wind_L, self.flaps.e)
            frame = self.sens.sample(t, x, f_b, self.eng.chamber_pressure(), False)
            cmd, tlm = self.ctrl.step(frame)
            if k % self.log_every == 0:
                for key, val in (("t", t), ("r", x[0:3].copy()), ("v", x[3:6].copy()),
                                 ("q", x[6:10].copy()), ("w", x[10:13].copy()), ("m", x[13]),
                                 ("T", self.eng.T.copy()), ("gimbal", self.eng.gimbal.copy()),
                                 ("flaps", self.flaps.e.copy()), ("wind", wind_L.copy()),
                                 ("qbar", info[0]), ("mach", info[1]), ("alpha", info[2])):
                    logs[key].append(val)
                for key, val in tlm.items():
                    tlm_log.setdefault(key, []).append(val)
            done = False
            for _ in range(sc.substeps):
                prop_left = x[13] - sc.veh.m_dry
                self.eng.step(h, cmd.engine_enable, cmd.throttle, cmd.gimbal, prop_left)
                self.flaps.step(h, cmd.flaps)
                self.rcs.step(h, cmd.rcs)
                vrel = np.linalg.norm(x[3:6] - wind_L)
                self.wind.step(h, x[2], vrel)
                wind_L = self.wind.total(t, x[2])
                F_e, M_e = self.eng.forces()
                x, _, _ = self.body.rk4(x, h, F_e, M_e + self.rcs.tau, self.eng.mdot(), wind_L, self.flaps.e)
                leg, hull = self._contact(x)
                if hull and not leg:
                    outcome, done = "crash", True
                    break
                if leg:
                    td = self._touchdown(t + h, x)
                    outcome, done = td["outcome"], True
                    break
            if done:
                break
        self.x = x
        out = {k: np.array(v) for k, v in logs.items()}
        out["tlm"] = {k: np.array(v) for k, v in tlm_log.items()}
        out["outcome"] = outcome
        out["touchdown"] = td
        out["final_state"] = x.copy()
        out["prop_used"] = sc.veh.m_wet - x[13]
        out["engine_starts"] = self.eng.n_starts.copy()
        return out

    def _touchdown(self, t, x) -> dict:
        sc = self.sc
        v = x[3:6]
        vz, vh = v[2], float(np.linalg.norm(v[:2]))
        tilt = quat.tilt_deg(x[6:10], 0)
        rate = math.degrees(float(np.linalg.norm(x[10:13])))
        d_pad = float(np.linalg.norm(x[0:2] - np.asarray(sc.pad[:2])))
        d_bak = float(np.linalg.norm(x[0:2] - np.asarray(sc.backup[:2])))
        soft = (-vz <= TD_MAX_VZ and vh <= TD_MAX_VH and tilt <= TD_MAX_TILT_DEG
                and rate <= TD_MAX_RATE_DEG_S)
        if soft and d_pad <= sc.pad_radius:
            outcome = "landed_pad"
        elif soft and d_bak <= 30.0:
            outcome = "landed_backup"
        elif soft:
            outcome = "landed_offsite"
        else:
            outcome = "hard_landing"
        return dict(t=t, r=x[0:3].copy(), v=v.copy(), vz=vz, vh=vh, tilt=tilt, rate=rate,
                    d_pad=d_pad, d_backup=d_bak, outcome=outcome, m=x[13])


def air_data(sc: Scenario, x, wind_L):
    """Convenience: (rho, a, airspeed_b) for diagnostics."""
    _, _, rho, a = us76(x[2], sc.env.rho_scale, sc.env.temp_offset)
    R = quat.to_dcm(x[6:10])
    return rho, a, R.T @ (x[3:6] - wind_L)
