"""
Point-mass landing-burn harness for guidance-law comparison.

Same dynamics as the G-FOLD reference (constant gravity minus a constant predicted aero
acceleration, rocket equation, thrust annulus [T_MIN, T_MAX], 25 deg tilt cone), so the
only difference between laws is the law itself. ZEM/ZEV and the neural policy are the
C flight-software functions (via ctypes); their commands pass through the same C
thrust_project() the executive uses.
"""
from __future__ import annotations

import numpy as np

import cguid
from gfold import ALPHA, GATE_R, GATE_V, T_MAX, T_MIN, TILT_MAX
from sim.params import G_EARTH

DT = 0.02
NN_HANDOVER_T = 3.0     # s to gate at which the learned policy hands over to ZEM/ZEV


def _handover(r, v):
    """Same rule as fsw.c: time-to-gate below NN_HANDOVER_T, or the policy stopped descending."""
    h = r[2] - GATE_R[2]
    return h / max(-v[2], 1e-3) < NN_HANDOVER_T or (v[2] > -1.0 and h > 5.0)


def sample_states(rng, n):
    """Burn-start states (post-flip) spanning the Monte Carlo envelope."""
    r = np.column_stack([rng.normal(0, 30, n), rng.normal(0, 30, n), rng.uniform(260, 460, n)])
    v = np.column_stack([rng.normal(0, 6, n), rng.normal(0, 6, n), rng.uniform(-55, -32, n)])
    m = rng.uniform(125e3, 130e3, n)
    aero = rng.uniform(0.0, 2.5, n)          # predicted vertical aero accel (tail-first drag)
    return r, v, m, aero


def _step(r, v, m, F, a_aero_z):
    acc = F / m + np.array([0.0, 0.0, -G_EARTH + a_aero_z])
    r = r + v * DT + 0.5 * acc * DT * DT
    v = v + acc * DT
    m = m - np.linalg.norm(F) * ALPHA * DT
    return r, v, m


def rollout(law, r0, v0, m0, a_aero_z, gamma=6.0, record=False, t_max=40.0):
    """Fly a guidance law from a burn-start state to the gate altitude.
    law: 'zemzev', 'nn' (pure learned) or 'hybrid' (learned, then ZEM/ZEV). Returns dict(fuel, dr_h, dv, t, states?)."""
    L = cguid.lib()
    r, v, m = np.array(r0, float), np.array(v0, float), float(m0)
    a_drift = np.array([0.0, 0.0, -G_EARTH + a_aero_z])
    g = None
    if law == "zemzev":
        g = cguid.Pdg()
        L.pdg_init(g, gamma, cguid.F3(*GATE_R), cguid.F3(*GATE_V))
    states, t = [], 0.0
    while t < t_max:
        if law == "hybrid" and g is None and _handover(r, v):
            # learned policy -> verified feedback law for the final seconds
            g = cguid.Pdg()
            L.pdg_init(g, gamma, cguid.F3(*GATE_R), cguid.F3(*GATE_V))
        if law == "zemzev" or (law == "hybrid" and g is not None):
            L.pdg_step(g, cguid.F3(*r), cguid.F3(*v), cguid.F3(*a_drift), DT)
            a_cmd = np.array(g.a_cmd[:])
        else:
            a_cmd = cguid.nn_accel(r - GATE_R, v, a_drift, m)
        F = cguid.thrust_project(m * a_cmd, T_MIN, T_MAX, TILT_MAX)
        if record:
            states.append((t, r.copy(), v.copy(), m))
        r_new, v_new, m_new = _step(r, v, m, F, a_aero_z)
        t += DT
        done = r_new[2] <= GATE_R[2] or (g is not None and g.tgo <= 0.31)
        r, v, m = r_new, v_new, m_new
        if done:
            break
    if g is not None and g.tgo > 0.0 and r[2] > GATE_R[2]:
        # last 0.3 s with the final command held (gains singular at tgo -> 0)
        n = int(round(g.tgo / DT))
        for _ in range(n):
            F = cguid.thrust_project(m * np.array(g.a_cmd[:]), T_MIN, T_MAX, TILT_MAX)
            r, v, m = _step(r, v, m, F, a_aero_z)
            t += DT
    out = dict(fuel=m0 - m, dr_h=float(np.linalg.norm(r[:2] - GATE_R[:2])),
               dz=float(r[2] - GATE_R[2]), dv=float(np.linalg.norm(v - GATE_V)), t=t)
    if record:
        out["states"] = states
    return out
