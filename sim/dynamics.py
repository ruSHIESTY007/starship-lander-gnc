"""
6-DOF rigid-body equations of motion over a flat, non-rotating Earth, integrated with RK4.

State x = [r_L(3), v_L(3), q_LB(4), omega_B(3), m]. Actuator forces/moments and the wind
vector are held constant across one integration step (zero-order hold at the truth rate);
aerodynamics are re-evaluated at every RK4 stage because they depend on the state.
"""
from __future__ import annotations

import numpy as np

from . import quat
from .aero import AeroModel, cross_n
from .atmosphere import us76
from .params import EnvParams, G_EARTH, VehicleParams

GRAV = np.array([0.0, 0.0, -G_EARTH])


class RigidBody:
    def __init__(self, veh: VehicleParams, env: EnvParams):
        self.veh = veh
        self.env = env
        self.aero = AeroModel(veh)
        self.I = np.diag(veh.inertia)
        self.I_inv = np.diag(1.0 / np.asarray(veh.inertia))

    def deriv(self, x, F_b_act, M_b_act, mdot, wind_L, flap_e):
        r, v, q, w, m = x[0:3], x[3:6], x[6:10], x[10:13], x[13]
        R = quat.to_dcm(q)
        _, _, rho, a = us76(r[2], self.env.rho_scale, self.env.temp_offset)
        v_air_b = R.T @ (v - wind_L)
        F_aero, M_aero, info = self.aero.forces(v_air_b, w, rho, a, flap_e)
        F_b = F_aero + F_b_act
        M_b = M_aero + M_b_act
        mm = max(m, 1.0)
        acc = R @ F_b / mm + GRAV
        qd = 0.5 * quat.mul(q, np.array([0.0, *w]))
        wd = self.I_inv @ (M_b - cross_n(w, self.I @ w))
        xd = np.empty(14)
        xd[0:3] = v
        xd[3:6] = acc
        xd[6:10] = qd
        xd[10:13] = wd
        xd[13] = mdot
        return xd, F_b / mm, info

    def rk4(self, x, h, F_b_act, M_b_act, mdot, wind_L, flap_e):
        k1, f_sp, info = self.deriv(x, F_b_act, M_b_act, mdot, wind_L, flap_e)
        k2, _, _ = self.deriv(x + 0.5 * h * k1, F_b_act, M_b_act, mdot, wind_L, flap_e)
        k3, _, _ = self.deriv(x + 0.5 * h * k2, F_b_act, M_b_act, mdot, wind_L, flap_e)
        k4, _, _ = self.deriv(x + h * k3, F_b_act, M_b_act, mdot, wind_L, flap_e)
        xn = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[6:10] = quat.normalize(xn[6:10])
        return xn, f_sp, info
