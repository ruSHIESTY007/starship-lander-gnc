"""ctypes access to the flight-software guidance primitives (pdg_*, thrust_project,
nn_policy_*) for point-mass guidance studies -- the exact C code that flies."""
import ctypes as C

import numpy as np

from sim.fsw_bindings import load_lib

f = C.c_float
F3 = f * 3


class Pdg(C.Structure):
    _fields_ = [("gamma", f), ("tgo", f), ("tgo_freeze", f), ("r_tgt", F3), ("v_tgt", F3),
                ("a_cmd", F3), ("initialized", C.c_int32)]


_lib = None


def lib():
    global _lib
    if _lib is None:
        L = load_lib()
        L.pdg_init.argtypes = [C.POINTER(Pdg), f, F3, F3]
        L.pdg_step.argtypes = [C.POINTER(Pdg), F3, F3, F3, f]
        L.thrust_project.argtypes = [F3, f, f, f, F3]
        L.thrust_project.restype = C.c_int
        L.nn_policy_accel.argtypes = [F3, F3, F3, f, F3]
        L.nn_policy_accel.restype = C.c_int32
        L.nn_policy_unload.restype = None
        _lib = L
    return _lib


def thrust_project(F, Tmin, Tmax, tilt):
    out = F3()
    lib().thrust_project(F3(*F), Tmin, Tmax, tilt, out)
    return np.array(out[:])


def load_nn(params):
    arr = (f * len(params))(*np.asarray(params, np.float32).tolist())
    rc = lib().nn_policy_load(arr, len(params))
    if rc != 0:
        raise RuntimeError(f"nn_policy_load rc={rc}")


def nn_accel(r_rel, v, a_drift, mass):
    out = F3()
    if lib().nn_policy_accel(F3(*r_rel), F3(*v), F3(*a_drift), mass, out) != 0:
        raise RuntimeError("no policy loaded")
    return np.array(out[:])
