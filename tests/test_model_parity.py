"""The onboard (C, float32) atmosphere and aero models must match the truth models at
nominal parameters -- otherwise wind estimation and guidance feedforward are biased by
construction rather than by the dispersions we intend to study."""
import ctypes as C
from pathlib import Path

import numpy as np
import pytest

from sim.aero import AeroModel
from sim.atmosphere import us76
from sim.params import VehicleParams

LIB = Path(__file__).resolve().parents[1] / "fsw" / "build" / "libfsw.so"
F3 = C.c_float * 3
F4 = C.c_float * 4


@pytest.fixture(scope="module")
def lib():
    if not LIB.exists():
        pytest.skip("build fsw first (make fsw)")
    L = C.CDLL(str(LIB))
    L.atmo_us76.argtypes = [C.c_float, C.POINTER(C.c_float), C.POINTER(C.c_float)]
    L.aero_forces.restype = C.c_float
    L.aero_forces.argtypes = [F3, F3, C.c_float, C.c_float, F4, C.c_float, F3, F3]
    return L


@pytest.mark.parametrize("h", [0.0, 1500.0, 6000.0, 10999.0, 12000.0, 25000.0])
def test_atmosphere_parity(lib, h):
    rho, a = C.c_float(), C.c_float()
    lib.atmo_us76(h, C.byref(rho), C.byref(a))
    _, _, rho_t, a_t = us76(h)
    assert rho.value == pytest.approx(rho_t, rel=2e-5)
    assert a.value == pytest.approx(a_t, rel=2e-6)


def test_aero_parity(lib):
    am = AeroModel(VehicleParams())
    rng = np.random.default_rng(3)
    for _ in range(200):
        v = rng.normal(0, 70, 3)
        w = rng.normal(0, 0.2, 3)
        e = rng.uniform(0, 1, 4)
        _, _, rho, a = us76(rng.uniform(0, 12000))
        F, M = F3(), F3()
        lib.aero_forces(F3(*v), F3(*w), rho, a, F4(*e), 1.0, F, M)
        Ft, Mt, _ = am.forces(v, w, rho, a, e)
        scaleF = np.abs(Ft).max() + 1.0
        scaleM = np.abs(Mt).max() + 1.0
        assert np.abs(np.array(F[:]) - Ft).max() / scaleF < 1e-5
        assert np.abs(np.array(M[:]) - Mt).max() / scaleM < 1e-5
