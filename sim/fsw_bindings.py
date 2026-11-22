"""
ctypes bindings to the C flight software (fsw/build/libfsw.so) and a Controller adapter
that plugs it into the truth simulation.

The C library keeps its state in statics, so one process hosts one flight software
instance; Monte Carlo runs use one worker process per concurrent run.
"""
from __future__ import annotations

import ctypes as C
import math
from pathlib import Path

import numpy as np

from . import quat
from .params import Scenario
from .sensors import SensorFrame
from .simulation import ActuatorCmd

LIB_PATH = Path(__file__).resolve().parents[1] / "fsw" / "build" / "libfsw.so"
N_ENG, N_FLAP, NAV_NX, NAV_N_SENS = 3, 4, 18, 3
f = C.c_float
i32 = C.c_int32

MODE_NAMES = {0: "INIT", 1: "GLIDE", 2: "FLIP", 3: "BURN", 4: "TERMINAL", 5: "LANDED", 6: "ABORT_NO_THRUST"}
GUID_ZEMZEV, GUID_NEURAL = 0, 1


class FswInput(C.Structure):
    _fields_ = [("t", f), ("gyro", f * 3), ("accel", f * 3),
                ("gnss_valid", i32), ("gnss_pos", f * 3), ("gnss_vel", f * 3),
                ("radar_valid", i32), ("radar_antenna", i32), ("radar_range", f),
                ("pc", f * N_ENG), ("eng_ready", i32 * N_ENG), ("touchdown", i32)]


class FswCmd(C.Structure):
    _fields_ = [("engine_enable", i32 * N_ENG), ("throttle", f * N_ENG), ("gimbal", f * 2),
                ("flaps", f * N_FLAP), ("rcs", f * 3)]


class FswInit(C.Structure):
    _fields_ = [("p0", f * 3), ("v0", f * 3), ("q0", f * 4),
                ("sig_pos", f), ("sig_vel", f), ("sig_att", f),
                ("pad", f * 3), ("backup", f * 3), ("mass0", f), ("gamma", f),
                ("h_gate", f), ("v_gate", f), ("use_wind_ff", i32), ("guidance_law", i32)]


class FswTlm(C.Structure):
    _fields_ = [("t", f), ("mode", i32), ("target_id", i32),
                ("nav_p", f * 3), ("nav_v", f * 3), ("nav_q", f * 4), ("nav_bg", f * 3), ("nav_ba", f * 3),
                ("nav_sig", f * NAV_NX), ("nis", f * NAV_N_SENS), ("sens_failed", i32 * NAV_N_SENS),
                ("wind_est", f * 2), ("wind_sig", f * 2), ("wind_k", f), ("mass_est", f),
                ("q_des", f * 4), ("att_err", f * 3), ("F_des", f * 3), ("a_cmd", f * 3), ("a_aero", f * 3),
                ("tgo", f), ("h_ign", f), ("mu_abort", f), ("target", f * 3),
                ("n_eng_healthy", i32), ("n_eng_active", i32), ("eng_failed", i32 * N_ENG),
                ("alloc_saturated", i32), ("T_cmd", f), ("tau_req", f * 3), ("divert_priority", i32)]


_lib = None


def load_lib():
    global _lib
    if _lib is None:
        if not LIB_PATH.exists():
            raise FileNotFoundError(f"{LIB_PATH} not found -- run `make fsw`")
        _lib = C.CDLL(str(LIB_PATH))
        _lib.fsw_init.argtypes = [C.POINTER(FswInit)]
        _lib.fsw_step.argtypes = [C.POINTER(FswInput), C.POINTER(FswCmd), C.POINTER(FswTlm)]
        _lib.nn_policy_load.argtypes = [C.POINTER(f), i32]
        _lib.nn_policy_load.restype = i32
    return _lib


# telemetry fields copied into the per-frame log (name -> struct attribute)
_TLM_VEC = ("nav_p", "nav_v", "nav_q", "nav_ba", "nav_sig", "nis", "wind_est", "wind_sig",
            "q_des", "att_err", "F_des", "a_cmd", "a_aero", "target")
_TLM_SCAL = ("mode", "target_id", "wind_k", "mass_est", "tgo", "h_ign", "mu_abort",
             "n_eng_healthy", "n_eng_active", "alloc_saturated", "T_cmd", "divert_priority")


class FlightSoftware:
    """Controller adapter: SensorFrame -> C fsw_step -> ActuatorCmd."""

    def __init__(self, sc: Scenario, *, gamma: float = 6.0, use_wind_ff: bool = True,
                 guidance_law: int = GUID_ZEMZEV, nn_params: np.ndarray | None = None,
                 nav_err_sigma=(10.0, 0.5, math.radians(0.5)), h_gate: float = 40.0,
                 v_gate: float = 3.0, rng: np.random.Generator | None = None):
        self.lib = load_lib()
        rng = rng if rng is not None else np.random.default_rng(sc.seed + 7919)
        from .simulation import initial_state
        x0 = initial_state(sc)
        sp, sv, sa = nav_err_sigma
        q0 = quat.mul(x0[6:10], quat.from_rotvec(rng.normal(0, sa, 3)))
        ini = FswInit()
        ini.p0[:] = (x0[0:3] + rng.normal(0, sp, 3)).tolist()
        ini.v0[:] = (x0[3:6] + rng.normal(0, sv, 3)).tolist()
        ini.q0[:] = q0.tolist()
        ini.sig_pos, ini.sig_vel, ini.sig_att = sp, sv, sa
        ini.pad[:] = list(sc.pad)
        ini.backup[:] = list(sc.backup)
        ini.mass0 = 130000.0          # FSW believes the nominal loaded mass
        ini.gamma = gamma
        ini.h_gate, ini.v_gate = h_gate, v_gate
        ini.use_wind_ff = int(use_wind_ff)
        ini.guidance_law = int(guidance_law)
        if guidance_law == GUID_NEURAL:
            if nn_params is None:
                raise ValueError("neural guidance requested without parameters")
            arr = (f * len(nn_params))(*np.asarray(nn_params, np.float32).tolist())
            rc = self.lib.nn_policy_load(arr, len(nn_params))
            if rc != 0:
                raise RuntimeError(f"nn_policy_load failed ({rc})")
        self.lib.fsw_init(C.byref(ini))
        self.inp, self.cmd, self.tlm = FswInput(), FswCmd(), FswTlm()

    def step(self, fr: SensorFrame):
        inp = self.inp
        inp.t = fr.t
        inp.gyro[:] = fr.gyro.tolist()
        inp.accel[:] = fr.accel.tolist()
        inp.gnss_valid = int(fr.gnss_valid)
        if fr.gnss_valid:
            inp.gnss_pos[:] = fr.gnss_pos.tolist()
            inp.gnss_vel[:] = fr.gnss_vel.tolist()
        inp.radar_valid = int(fr.radar_valid)
        inp.radar_antenna = fr.radar_antenna
        inp.radar_range = fr.radar_range
        inp.pc[:] = fr.pc.tolist()
        inp.eng_ready[:] = [int(b) for b in fr.eng_ready]
        inp.touchdown = int(fr.touchdown)
        self.lib.fsw_step(C.byref(inp), C.byref(self.cmd), C.byref(self.tlm))
        c = self.cmd
        out = ActuatorCmd(engine_enable=np.array(c.engine_enable[:], bool),
                          throttle=np.array(c.throttle[:]), gimbal=np.array(c.gimbal[:]),
                          flaps=np.array(c.flaps[:]), rcs=np.array(c.rcs[:]))
        t = self.tlm
        tlm = {k: np.array(getattr(t, k)[:]) for k in _TLM_VEC}
        tlm.update({k: getattr(t, k) for k in _TLM_SCAL})
        tlm["eng_failed"] = np.array(t.eng_failed[:])
        tlm["sens_failed"] = np.array(t.sens_failed[:])
        return out, tlm
