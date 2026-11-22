"""
Offline fuel-optimal landing-burn reference (3-DOF) via lossless convexification.

Formulation: Acikmese & Ploen (2007), Blackmore, Acikmese & Scharf (2010) ("G-FOLD").
With u = T/m, sigma = Gamma/m and z = ln m, the non-convex thrust annulus
rho1 <= |T| <= rho2 becomes a second-order cone program whose relaxation is provably
tight. Time of flight is found by a coarse grid + golden-section search over SOCP solves.

Problem solved here (the landing burn from post-flip state to the terminal gate):
  * constant gravity, no aerodynamics (the comparison harness uses the same model)
  * thrust annulus [1 engine at min throttle, 2 engines at max] -- this *relaxes* the
    engine-count logic (any thrust in the band is allowed), so the optimum is a
    lower bound on the propellant any 1-2 engine strategy can achieve
  * thrust tilt <= 25 deg from vertical (same cone the flight software enforces)
  * final state = terminal gate (position + velocity)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.params import G0, G_EARTH, VehicleParams  # noqa: E402

VEH = VehicleParams()
T_MIN = VEH.throttle_min * VEH.T_max_each          # one engine at minimum throttle
T_MAX = 2.0 * VEH.T_max_each                        # two engines at full throttle
TILT_MAX = np.deg2rad(25.0)
ALPHA = 1.0 / (VEH.isp * G0)
GATE_R = np.array([0.0, 0.0, 24.0 + 40.0])          # CoM at gate (legs 24 m + 40 m)
GATE_V = np.array([0.0, 0.0, -3.0])


@dataclass
class GfoldResult:
    tf: float
    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    m: np.ndarray
    u: np.ndarray          # thrust acceleration T/m (N+1, 3)
    fuel: float


def solve_fixed_tf(r0, v0, m0, tf, N=40, rf=GATE_R, vf=GATE_V, a_aero_z=0.0):
    """a_aero_z: constant predicted vertical aero acceleration (same drift model as ZEM/ZEV)."""
    g = np.array([0.0, 0.0, -G_EARTH + a_aero_z])
    dt = tf / N
    tk = np.arange(N + 1) * dt
    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    z = cp.Variable(N + 1)
    u = cp.Variable((N + 1, 3))
    s = cp.Variable(N + 1)
    z0 = np.log(np.maximum(m0 - ALPHA * T_MAX * tk, VEH.m_dry))
    mu1, mu2 = T_MIN * np.exp(-z0), T_MAX * np.exp(-z0)
    cons = [r[0] == r0, v[0] == v0, z[0] == np.log(m0), r[N] == rf, v[N] == vf,
            z[N] >= np.log(VEH.m_dry)]
    # first-order-hold dynamics (exact for piecewise-linear u)
    cons += [v[1:] == v[:-1] + dt * (u[:-1] + u[1:]) / 2 + g * dt,
             r[1:] == r[:-1] + v[:-1] * dt + dt ** 2 * (u[:-1] / 3 + u[1:] / 6) + 0.5 * g * dt ** 2,
             z[1:] == z[:-1] - ALPHA * dt * (s[:-1] + s[1:]) / 2]
    dz = z - z0
    cons += [cp.norm(u, axis=1) <= s,
             cp.multiply(mu1, 1 - dz + cp.square(dz) / 2) <= s,
             s <= cp.multiply(mu2, 1 - dz),
             z >= z0,
             z <= np.log(np.maximum(m0 - ALPHA * T_MIN * tk, 1.0)),
             u[:, 2] >= np.cos(TILT_MAX) * s,
             r[:, 2] >= rf[2] - 1.0]
    prob = cp.Problem(cp.Maximize(z[N]), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate") or z.value is None:
        return None
    m = np.exp(z.value)
    return GfoldResult(tf, tk, r.value, v.value, m, u.value, m0 - m[-1])


def solve(r0, v0, m0, t_lo=3.0, t_hi=40.0, tol=0.05, N=40, **kw):
    """Fuel-optimal landing burn with free final time (grid + golden section)."""
    cache: dict[float, GfoldResult | None] = {}

    def fuel(tf):
        tf = round(float(tf), 4)
        if tf not in cache:
            cache[tf] = solve_fixed_tf(r0, v0, m0, tf, N=N, **kw)
        return np.inf if cache[tf] is None else cache[tf].fuel

    grid = np.linspace(t_lo, t_hi, 16)
    vals = np.array([fuel(t) for t in grid])
    if not np.isfinite(vals).any():
        return None
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    while b - a > tol:
        if fuel(c) < fuel(d):
            b, d = d, c
            c = b - phi * (b - a)
        else:
            a, c = c, d
            d = a + phi * (b - a)
    feas = [k for k, val in cache.items() if val is not None]
    return cache[min(feas, key=lambda k: cache[k].fuel)]


if __name__ == "__main__":
    import time
    r0, v0, m0 = np.array([-30.0, 20.0, 330.0]), np.array([2.0, -1.0, -42.0]), 128000.0
    t0 = time.time()
    res = solve(r0, v0, m0)
    print(f"G-FOLD: tf={res.tf:.2f} s fuel={res.fuel:.0f} kg  ({time.time() - t0:.1f} s wall)")
    T = np.linalg.norm(res.u, axis=1) * res.m
    print(f"  thrust {T.min() / 1e6:.2f}..{T.max() / 1e6:.2f} MN (bounds {T_MIN / 1e6:.2f}..{T_MAX / 1e6:.2f})")
    print("  thrust profile (MN):", np.round(T[::4] / 1e6, 2))
