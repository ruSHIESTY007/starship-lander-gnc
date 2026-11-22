"""Physics verification of the truth simulation (analytic and conservation checks)."""
import math

import numpy as np
import pytest

from sim import quat
from sim.actuators import Engines
from sim.aero import AeroModel
from sim.atmosphere import us76
from sim.dynamics import RigidBody
from sim.params import G0, EnvParams, Scenario, SensorParams, FaultParams, VehicleParams
from sim.sensors import SensorSuite
from sim.wind import WindModel


def test_us76_reference_points():
    T, p, rho, a = us76(0.0)
    assert T == pytest.approx(288.15) and p == pytest.approx(101325.0) and rho == pytest.approx(1.2250, rel=1e-4)
    assert a == pytest.approx(340.29, rel=1e-3)
    # USSA-76 tables (geometric altitude)
    assert us76(5000.0)[2] == pytest.approx(0.73643, rel=2e-3)
    assert us76(10000.0)[2] == pytest.approx(0.41351, rel=2e-3)
    assert us76(15000.0)[2] == pytest.approx(0.19476, rel=3e-3)
    assert us76(11000.0)[0] == pytest.approx(216.77, abs=0.2)


def _vacuum_body():
    return RigidBody(VehicleParams(), EnvParams(rho_scale=0.0))


def test_vacuum_free_fall_matches_analytic():
    body = _vacuum_body()
    x = np.zeros(14); x[0:3] = (10.0, -5.0, 1000.0); x[3:6] = (3.0, 1.0, -20.0); x[6] = 1.0; x[13] = 1e5
    h, n = 0.01, 500
    for _ in range(n):
        x, _, _ = body.rk4(x, h, np.zeros(3), np.zeros(3), 0.0, np.zeros(3), np.zeros(4))
    t = h * n
    assert x[2] == pytest.approx(1000.0 - 20.0 * t - 0.5 * G0 * t * t, abs=1e-6)
    assert x[0] == pytest.approx(10.0 + 3.0 * t, abs=1e-9)


def test_torque_free_tumble_conserves_momentum_and_energy():
    body = _vacuum_body()
    I = body.I
    x = np.zeros(14); x[2] = 1e5; x[6:10] = quat.from_euler321(0.3, -0.2, 0.9); x[10:13] = (0.4, 0.05, -0.1); x[13] = 1e5
    def H_L(x):
        return quat.to_dcm(x[6:10]) @ (I @ x[10:13])
    H0, E0 = H_L(x), 0.5 * x[10:13] @ I @ x[10:13]
    for _ in range(2000):
        x, _, _ = body.rk4(x, 0.005, np.zeros(3), np.zeros(3), 0.0, np.zeros(3), np.zeros(4))
    assert np.linalg.norm(H_L(x) - H0) / np.linalg.norm(H0) < 1e-6
    assert 0.5 * x[10:13] @ I @ x[10:13] == pytest.approx(E0, rel=1e-6)


def test_belly_flop_terminal_velocity_force_balance():
    veh = VehicleParams()
    am = AeroModel(veh)
    _, _, rho, a = us76(0.0)
    e = np.zeros(4)                                  # flaps retracted: body strips only
    s_total = 5 * veh.strip_area
    v_t = math.sqrt(2 * veh.m_wet * G0 / (rho * 1.20 * s_total))
    F, M, _ = am.forces(np.array([0.0, 0.0, v_t]), np.zeros(3), rho, a, e)
    assert -F[2] == pytest.approx(veh.m_wet * G0, rel=1e-9)
    # centre of pressure is ahead of the CoM -> nose-up (unstable) moment in belly-flop
    assert M[1] > 0


def test_crossflow_rotation_is_damped():
    am = AeroModel(VehicleParams())
    _, _, rho, a = us76(3000.0)
    v = np.array([0.0, 0.0, 80.0])
    _, M0, _ = am.forces(v, np.zeros(3), rho, a, np.full(4, 0.5))
    _, M1, _ = am.forces(v, np.array([0.0, 0.1, 0.0]), rho, a, np.full(4, 0.5))
    _, M2, _ = am.forces(v, np.array([0.0, 0.0, 0.1]), rho, a, np.full(4, 0.5))
    assert M1[1] - M0[1] < 0          # pitch damping
    assert M2[2] - M0[2] < 0 or abs(M2[2] - M0[2]) < 1e3   # yaw: no crossflow change at alpha=90


def test_engine_mass_flow_spool_and_failure():
    veh = VehicleParams()
    eng = Engines(veh)
    dt = 0.005
    for _ in range(int(veh.spool_up_time / dt) + 400):
        eng.step(dt, [True] * 3, [1.0] * 3, [0.0, 0.0], 1e4)
    assert eng.T.sum() == pytest.approx(3 * veh.T_max_each, rel=1e-3)
    assert eng.mdot() == pytest.approx(-3 * veh.T_max_each / (veh.isp * G0), rel=1e-3)
    eng.fail(1)
    for _ in range(200):
        eng.step(dt, [True] * 3, [1.0] * 3, [0.0, 0.0], 1e4)
    assert eng.T[1] < 1e-3 * veh.T_max_each


def test_gimbal_torque_sign_convention():
    veh = VehicleParams()
    eng = Engines(veh)
    for _ in range(1000):
        eng.step(0.005, [True] * 3, [0.6] * 3, [0.0, 0.1], 1e4)
    F, M = eng.forces()
    # positive z-lateral thrust at the aft end -> positive pitch (about +y_B) moment
    assert M[1] > 0 and M[1] == pytest.approx(-veh.x_aft * F[2], rel=1e-2)
    for _ in range(1000):
        eng.step(0.005, [True] * 3, [0.6] * 3, [0.1, 0.0], 1e4)
    F, M = eng.forces()
    assert M[2] < 0


def test_wind_profile_and_turbulence_statistics():
    env = EnvParams(wind_surface=(5.0, 0.0), wind_aloft=(20.0, 0.0), turb_sigma=2.0)
    wm = WindModel(env, np.random.default_rng(0))
    assert wm.mean(10.0)[0] == pytest.approx(5.0)
    assert wm.mean(10000.0)[0] == pytest.approx(20.0)
    assert wm.mean(500.0)[0] > wm.mean(50.0)[0]
    samples = []
    for _ in range(40000):
        wm.step(0.05, 5000.0, 100.0)
        samples.append(wm.turb.copy())
    s = np.array(samples)
    assert s.std(axis=0) == pytest.approx([2.0, 2.0, 2.0], rel=0.15)


def test_radar_slant_range_and_antenna_selection():
    sp = SensorParams(radar_sigma_abs=0.0, radar_sigma_rel=0.0)
    ss = SensorSuite(sp, FaultParams(), np.random.default_rng(0), 0.01)
    x = np.zeros(14); x[2] = 1000.0
    x[6:10] = quat.from_euler321(0.0, 0.0, math.pi)          # belly down
    fr = ss.sample(0.0, x, np.zeros(3), np.zeros(3), False)
    assert fr.radar_valid and fr.radar_antenna == 0 and fr.radar_range == pytest.approx(1000.0)
    x[6:10] = quat.from_euler321(0.0, -math.pi / 2 + 0.2, 0.0)  # nose up, tilted 0.2 rad
    ss.k = 0
    fr = ss.sample(0.0, x, np.zeros(3), np.zeros(3), False)
    assert fr.radar_antenna == 1 and fr.radar_range == pytest.approx(1000.0 / math.cos(0.2), rel=1e-6)
