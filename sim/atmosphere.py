"""
U.S. Standard Atmosphere 1976, 0-47 km (geopotential altitude), with dispersion hooks.
"""
from __future__ import annotations

import math

R_AIR = 287.053
GAMMA = 1.4
G0 = 9.80665
# (base geopotential altitude [m], base temperature [K], lapse rate [K/m], base pressure [Pa])
_LAYERS = (
    (0.0,     288.15, -0.0065, 101325.0),
    (11000.0, 216.65,  0.0,     22632.06),
    (20000.0, 216.65,  0.001,   5474.889),
    (32000.0, 228.65,  0.0028,  868.0187),
)
R_EARTH = 6356766.0


def us76(h_geom: float, rho_scale: float = 1.0, temp_offset: float = 0.0):
    """Return (T [K], p [Pa], rho [kg/m^3], a [m/s]) at geometric altitude h_geom [m].

    rho_scale multiplies density (pressure consistent); temp_offset shifts T (affects a).
    Below sea level the troposphere law is extrapolated.
    """
    h = R_EARTH * h_geom / (R_EARTH + h_geom)          # geopotential altitude
    h = min(h, 47000.0)
    base = _LAYERS[0]
    for layer in _LAYERS:
        if h >= layer[0]:
            base = layer
    hb, Tb, L, pb = base
    T = Tb + L * (h - hb)
    if L == 0.0:
        p = pb * math.exp(-G0 * (h - hb) / (R_AIR * Tb))
    else:
        p = pb * (T / Tb) ** (-G0 / (L * R_AIR))
    rho = p / (R_AIR * T) * rho_scale
    T_eff = T + temp_offset
    a = math.sqrt(GAMMA * R_AIR * T_eff)
    return T_eff, p * rho_scale, rho, a
