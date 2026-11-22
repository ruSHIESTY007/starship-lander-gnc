"""
Aerodynamic model: slender-body crossflow strips + axial drag + four flat-plate flaps.

Crossflow theory (Allen & Perkins): each body strip sees the local velocity component
normal to the body axis and produces a drag force opposite that component,
    dF = -1/2 rho |v_c| v_c  Cdc(M_c)  dA.
Evaluating strips at their stations automatically yields the centre of pressure (and
therefore the static instability of the aft-heavy belly-flop attitude) and rotational
damping (omega x r adds to local crossflow). Lift in the glide is the component of the
normal force perpendicular to the airspeed when alpha != 90 deg.

The same model (nominal coefficients) is replicated in the flight software for wind
estimation and guidance feedforward; truth disperses cdc_scale / ca / rho.
"""
from __future__ import annotations

import math

import numpy as np

from .params import VehicleParams

_CDC_M = np.array([0.0, 0.4, 0.9, 1.2, 3.0])
_CDC_V = np.array([1.20, 1.20, 1.65, 1.50, 1.50])
_CA_M = np.array([0.0, 0.6, 1.0, 1.3, 3.0])
_CA_V = np.array([1.00, 1.00, 1.60, 1.40, 1.30])


def cdc(mach):
    return np.interp(mach, _CDC_M, _CDC_V)


def ca_factor(mach):
    return np.interp(mach, _CA_M, _CA_V)


def cross_n(a, b):
    """Row-wise cross product for (N,3) arrays (np.cross is ~10x slower for tiny arrays)."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


class AeroModel:
    def __init__(self, veh: VehicleParams):
        off = np.asarray(veh.com_offset, float)
        self.veh = veh
        self.r_strip = np.stack([np.asarray(veh.strip_x), np.zeros(5), np.zeros(5)], axis=1) - off
        self.r_flap = veh.flap_positions() - off
        self.n_flap = veh.flap_normals()
        self.a_strip = veh.strip_area
        self.a_flap = np.asarray(veh.flap_area, float)
        self.s_base = veh.base_area
        self.r_axial = -off   # axial force acts on the body centreline

    def forces(self, v_air_b, omega_b, rho, a_sound, flap_e):
        """Return (F_b, M_b, (qbar, mach, alpha_deg)) for vehicle velocity relative to air
        v_air_b (body frame), body rate omega_b and flap exposures flap_e (4,) in [0, 1].

        Written with explicit component algebra (all stations lie in the B x-y plane,
        offset by the CoM error) because tiny-array numpy calls dominate the runtime.
        """
        veh = self.veh
        u, v, w = float(v_air_b[0]), float(v_air_b[1]), float(v_air_b[2])
        wx, wy, wz = float(omega_b[0]), float(omega_b[1]), float(omega_b[2])
        # --- crossflow strips: r = (xs, ys, zs) ---
        xs, ys, zs = self.r_strip[:, 0], self.r_strip[:, 1], self.r_strip[:, 2]
        vy = v + (wz * xs - wx * zs)
        vz = w + (wx * ys - wy * xs)
        Vc = np.sqrt(vy * vy + vz * vz)
        k = -0.5 * rho * self.a_strip * veh.cdc_scale * Vc * np.interp(Vc / a_sound, _CDC_M, _CDC_V)
        Fy_s, Fz_s = k * vy, k * vz
        Fx = 0.0
        Fy = float(Fy_s.sum())
        Fz = float(Fz_s.sum())
        Mx = float((ys * Fz_s - zs * Fy_s).sum())
        My = float((-xs * Fz_s).sum())
        Mz = float((xs * Fy_s).sum())
        # --- axial (acts on the centreline) ---
        V = math.sqrt(u * u + v * v + w * w)
        mach = V / a_sound
        ca = (veh.ca_nose_first if u > 0 else veh.ca_tail_first) * float(np.interp(mach, _CA_M, _CA_V))
        Fa = -0.5 * rho * abs(u) * u * self.s_base * ca
        ra = self.r_axial
        Fx += Fa
        My += ra[2] * Fa
        Mz -= ra[1] * Fa
        # --- flaps: r = (xf, yf, zf), normal n = (0, ny, nz) ---
        xf, yf, zf = self.r_flap[:, 0], self.r_flap[:, 1], self.r_flap[:, 2]
        ny, nz = self.n_flap[:, 1], self.n_flap[:, 2]
        lvx = u + (wy * zf - wz * yf)
        lvy = v + (wz * xf - wx * zf)
        lvz = w + (wx * yf - wy * xf)
        del lvx
        vn = lvy * ny + lvz * nz
        mag = -0.5 * rho * np.abs(vn) * vn * self.a_flap * veh.flap_cf * np.clip(flap_e, 0.0, 1.0)
        fy, fz = mag * ny, mag * nz
        Fy += float(fy.sum())
        Fz += float(fz.sum())
        Mx += float((yf * fz - zf * fy).sum())
        My += float((-xf * fz).sum())
        Mz += float((xf * fy).sum())
        alpha = math.degrees(math.atan2(w, u))
        return np.array([Fx, Fy, Fz]), np.array([Mx, My, Mz]), (0.5 * rho * V * V, mach, alpha)
