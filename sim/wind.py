"""
Wind model: altitude-dependent mean profile + first-order (Dryden-like) turbulence + an
optional discrete 1-cosine gust.

Mean wind: 1/7 power law in the surface layer (to 1 km), then linear interpolation to the
aloft value at 10 km (constant above). This gives realistic shear and directional veer.

Turbulence: each axis is a first-order Gauss-Markov process in *distance travelled*
(Taylor frozen-turbulence hypothesis), the first-order approximation of the Dryden spectrum,
with MIL-HDBK-1797 low-altitude length scales L_w = h, L_u = L_v = h / (0.177 + 0.000823 h)^1.2
(h in ft, clamped).
"""
from __future__ import annotations

import math

import numpy as np

from .params import EnvParams

FT = 0.3048


class WindModel:
    def __init__(self, env: EnvParams, rng: np.random.Generator):
        self.env = env
        self.rng = rng
        self.turb = np.zeros(3)
        self.ws = np.array([*env.wind_surface, 0.0])
        self.wa = np.array([*env.wind_aloft, 0.0])

    def mean(self, h: float) -> np.ndarray:
        h = max(h, 0.0)
        w1k = self.ws * (1000.0 / 10.0) ** (1.0 / 7.0)
        if h <= 1000.0:
            return self.ws * (max(h, 0.5) / 10.0) ** (1.0 / 7.0)
        f = min((h - 1000.0) / 9000.0, 1.0)
        return (1.0 - f) * w1k + f * self.wa

    def step(self, dt: float, h: float, airspeed: float):
        """Advance turbulence state over dt at altitude h [m] with true airspeed [m/s]."""
        sig = self.env.turb_sigma
        if sig <= 0.0:
            self.turb[:] = 0.0
            return
        if h < 1000.0 * FT:
            hft = max(h / FT, 10.0)
            Lw = hft * FT
            Luv = hft / (0.177 + 0.000823 * hft) ** 1.2 * FT
            s_w = sig
            s_uv = sig / (0.177 + 0.000823 * hft) ** 0.4 * 0.35   # normalized so s_uv ~ sig at 1000 ft
            s_uv = min(max(s_uv, sig), 2.0 * sig)
        else:
            Lw = Luv = 1750.0 * FT
            s_w = s_uv = sig
        V = max(airspeed, 1.0)
        for i, (L, s) in enumerate(((Luv, s_uv), (Luv, s_uv), (Lw, s_w))):
            phi = math.exp(-V * dt / L)
            self.turb[i] = phi * self.turb[i] + s * math.sqrt(1.0 - phi * phi) * self.rng.standard_normal()

    def gust(self, t: float) -> np.ndarray:
        g = self.env.gust
        if g is None:
            return np.zeros(3)
        t0, dur, amp = g
        if t0 <= t <= t0 + dur:
            return 0.5 * (1 - math.cos(2 * math.pi * (t - t0) / dur)) * np.asarray(amp, float)
        return np.zeros(3)

    def total(self, t: float, h: float) -> np.ndarray:
        """Wind velocity in the L frame (air mass velocity)."""
        w = self.mean(h) + self.turb + self.gust(t)
        if h < 50.0:
            w = w * np.array([1.0, 1.0, max(h, 0.0) / 50.0])   # vertical gusts vanish at ground
        return w
