"""
Nominal closed-loop SIL flight: C flight software driving the Python 6-DOF truth model.
Produces docs/figures/nominal_*.png and prints a summary.
    .venv/bin/python analysis/run_nominal.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plotstyle as ps
from sim import quat
from sim.fsw_bindings import MODE_NAMES, FlightSoftware
from sim.params import Scenario
from sim.simulation import Simulation

FIG = ROOT / "docs" / "figures"


def shade_modes(ax, t, modes):
    edges = np.r_[0, np.nonzero(np.diff(modes))[0] + 1, len(t) - 1]
    for a, b in zip(edges[:-1], edges[1:]):
        ax.axvspan(t[a], t[b], color=ps.MODE_COLORS.get(int(modes[a]), "#eee"), lw=0, zorder=0)


def main(seed=0):
    ps.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    sc = Scenario(seed=seed)
    out = Simulation(sc, FlightSoftware(sc)).run()
    t, r, v, tl = out["t"], out["r"], out["v"], out["tlm"]
    modes = tl["mode"]
    td = out["touchdown"]
    print(f"outcome {out['outcome']}  miss {td['d_pad']:.2f} m  vz {td['vz']:.2f}  vh {td['vh']:.2f}  "
          f"tilt {td['tilt']:.2f} deg  prop {out['prop_used']:.0f} kg")
    tilt = np.array([quat.tilt_deg(q) for q in out["q"]])

    # ---------- figure 1: trajectory + timeline ----------
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = axs[0, 0]
    ax.plot(r[:, 0], r[:, 2] / 1000, color=ps.SERIES[0])
    k_ign = np.argmax(modes >= 2)
    ax.plot(r[k_ign, 0], r[k_ign, 2] / 1000, "o", ms=8, color=ps.SERIES[1], mec=ps.SURFACE, mew=2)
    ax.annotate("flip / ignition", (r[k_ign, 0], r[k_ign, 2] / 1000), xytext=(-150, 40),
                textcoords="offset points", fontsize=8, color=ps.INK2,
                arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8))
    ax.set(title="Downrange profile", xlabel="downrange x [m]", ylabel="altitude [km]")
    ax = axs[0, 1]
    seg = modes >= 2
    ax.plot(r[seg, 0], r[seg, 1], color=ps.SERIES[0], label="powered phase ground track")
    ax.plot(0, 0, "s", ms=8, color=ps.INK2)
    ax.add_patch(plt.Circle((0, 0), sc.pad_radius, fill=False, ec=ps.MUTED, lw=1, ls="--"))
    ax.plot(td["r"][0], td["r"][1], "o", ms=8, color=ps.SERIES[1], mec=ps.SURFACE, mew=2)
    ax.annotate(f"touchdown, {td['d_pad']:.2f} m from centre", (td["r"][0], td["r"][1]),
                xytext=(10, -25), textcoords="offset points", fontsize=8, color=ps.INK2)
    ax.set_aspect("equal")
    ax.set(title="Powered-phase ground track (pad radius 10 m dashed)", xlabel="x [m]", ylabel="y [m]")
    ax = axs[1, 0]
    shade_modes(ax, t, modes)
    ax.plot(t, -v[:, 2], color=ps.SERIES[0], label="descent rate")
    ax.plot(t, np.linalg.norm(v[:, :2], axis=1), color=ps.SERIES[1], label="horizontal speed")
    ax.set(title="Velocity (shading: glide / flip / burn / terminal)", xlabel="time [s]", ylabel="m/s")
    ax.legend(loc="upper right")
    ax = axs[1, 1]
    shade_modes(ax, t, modes)
    ax.plot(t, tilt, color=ps.SERIES[0], label="body-axis tilt from vertical")
    glide = modes <= 2
    ax.plot(t[glide], out["alpha"][glide], color=ps.SERIES[2], label="angle of attack (glide/flip)")
    ax.set(title="Attitude", xlabel="time [s]", ylabel="deg")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "nominal_trajectory.png")

    # ---------- figure 2: powered phase zoom ----------
    k0 = max(np.argmax(modes >= 2) - 200, 0)
    sl = slice(k0, len(t))
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2))
    ax = axs[0, 0]
    shade_modes(ax, t[sl], modes[sl])
    for i in range(3):
        ax.plot(t[sl], out["T"][sl, i] / 1e6, color=ps.SERIES[i], label=["centre engine", "outboard +y", "outboard -y"][i])
    ax.set(title="Engine thrust (3 lit for flip, 2 for burn, centre for terminal)", xlabel="time [s]", ylabel="MN")
    ax.legend(loc="upper right")
    ax = axs[0, 1]
    shade_modes(ax, t[sl], modes[sl])
    ax.plot(t[sl], r[sl, 2] - 24.0, color=ps.SERIES[0])
    ax.set(title="Leg height above ground", xlabel="time [s]", ylabel="m")
    ax = axs[1, 0]
    shade_modes(ax, t[sl], modes[sl])
    for i in range(4):
        ax.plot(t[sl], out["flaps"][sl, i], color=ps.SERIES[i], lw=1.2, label=["fore R", "fore L", "aft R", "aft L"][i])
    ax.set(title="Flap exposure (aerodynamic pre-flip, then retracted)", xlabel="time [s]", ylabel="fraction")
    ax.legend(loc="upper right", ncol=2)
    ax = axs[1, 1]
    shade_modes(ax, t[sl], modes[sl])
    ax.plot(t[sl], np.degrees(np.arcsin(np.clip(np.linalg.norm(out["gimbal"][sl], axis=1), 0, 1))), color=ps.SERIES[0])
    ax.set(title="Gimbal deflection magnitude", xlabel="time [s]", ylabel="deg")
    fig.tight_layout()
    fig.savefig(FIG / "nominal_powered.png")

    # ---------- figure 3: navigation & wind ----------
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2))
    err = tl["nav_p"] - r
    sig = tl["nav_sig"]
    for j, (ax, lab) in enumerate(zip([axs[0, 0], axs[0, 1]], ["x (East)", "z (Up)"])):
        i = 0 if j == 0 else 2
        ax.fill_between(t, -3 * sig[:, i], 3 * sig[:, i], color=ps.GRID, lw=0, label="±3σ filter")
        ax.plot(t, err[:, i], color=ps.SERIES[0], lw=1.0, label="estimate − truth")
        ax.set(title=f"Position error, {lab}", xlabel="time [s]", ylabel="m", ylim=(-8, 8))
        ax.legend(loc="upper right")
    ax = axs[1, 0]
    verr = tl["nav_v"] - v
    ax.fill_between(t, -3 * sig[:, 3], 3 * sig[:, 3], color=ps.GRID, lw=0, label="±3σ filter")
    ax.plot(t, verr[:, 0], color=ps.SERIES[0], lw=1.0, label="estimate − truth (x)")
    ax.set(title="Velocity error", xlabel="time [s]", ylabel="m/s", ylim=(-0.6, 0.6))
    ax.legend(loc="upper right")
    ax = axs[1, 1]
    ax.plot(t, out["wind"][:, 0], color=ps.SERIES[0], lw=0.8, alpha=0.5, label="true wind x (incl. turbulence)")
    ax.plot(t, tl["wind_est"][:, 0], color=ps.SERIES[0], lw=1.8, label="estimate x")
    ax.plot(t, out["wind"][:, 1], color=ps.SERIES[1], lw=0.8, alpha=0.5, label="true wind y")
    ax.plot(t, tl["wind_est"][:, 1], color=ps.SERIES[1], lw=1.8, label="estimate y")
    ax.set(title="Aero-inverting wind estimator (no air data)", xlabel="time [s]", ylabel="m/s")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "nominal_nav.png")
    plt.close("all")
    phases = {MODE_NAMES[int(m)]: float(t[np.argmax(modes == m)]) for m in np.unique(modes)}
    print("mode entry times:", phases)
    return out


if __name__ == "__main__":
    main()
