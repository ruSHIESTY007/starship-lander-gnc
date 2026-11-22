"""Summarise the Monte Carlo campaigns: tables (stdout + out/mc_summary.json) and figures."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plotstyle as ps

FIG = ROOT / "docs" / "figures"
OUTCOMES = ["landed_pad", "landed_backup", "landed_offsite", "hard_landing", "crash", "timeout"]


def load(name):
    p = ROOT / "out" / f"mc_{name}.json"
    return json.load(open(p)) if p.exists() else []


def stats(rows):
    if not rows:
        return {}
    n = len(rows)
    oc = {o: sum(r["outcome"] == o for r in rows) / n for o in OUTCOMES}
    soft = [r for r in rows if r["outcome"].startswith("landed")]
    pad = [r for r in rows if r["outcome"] == "landed_pad"]
    d = np.array([r["d_pad"] for r in pad]) if pad else np.array([np.nan])
    nerr = np.array([r["nav_nerr"] for r in rows])
    fail = [r for r in rows if r["outcome"] != "landed_pad"]
    taxonomy = dict(
        propellant_exhausted=sum(r["prop"] > 29000 for r in fail) / n,
        far_from_pad_gt100m=sum(r["prop"] <= 29000 and r["d_pad"] > 100 for r in fail) / n,
        other=sum(r["prop"] <= 29000 and r["d_pad"] <= 100 for r in fail) / n)
    pad_ok = [r for r in rows if r["outcome"] == "landed_pad"]
    w_ok = float(np.mean([r["wind_aloft"] for r in pad_ok])) if pad_ok else None
    w_fail = float(np.mean([r["wind_aloft"] for r in fail])) if fail else None
    return dict(n=n, outcomes=oc, failure_taxonomy=taxonomy, wind_aloft_mean_success=w_ok,
                wind_aloft_mean_failure=w_fail,
                soft_landing_rate=len(soft) / n,
                miss_median=float(np.median(d)), miss_p95=float(np.percentile(d, 95)), miss_max=float(np.max(d)),
                vz_p95=float(np.percentile([-r["vz"] for r in soft], 95)) if soft else None,
                vh_p95=float(np.percentile([r["vh"] for r in soft], 95)) if soft else None,
                tilt_p95=float(np.percentile([r["tilt"] for r in soft], 95)) if soft else None,
                prop_mean=float(np.mean([r["prop"] for r in rows])),
                prop_p95=float(np.percentile([r["prop"] for r in rows], 95)),
                nav_nees_pos_mean=float(np.mean(np.sum(nerr ** 2, axis=1))),
                wind_err_at_ign_median=float(np.median([r["wind_err_at_ign"] for r in rows])))


def main():
    ps.apply()
    camps = {k: load(k) for k in ("baseline", "no_wind_ff", "engine_out", "neural")}
    summ = {k: stats(v) for k, v in camps.items()}
    json.dump(summ, open(ROOT / "out" / "mc_summary.json", "w"), indent=2)
    for k, s in summ.items():
        if not s:
            continue
        print(f"\n== {k} (n={s['n']})")
        print("  outcomes: " + ", ".join(f"{o} {100 * p:.1f}%" for o, p in s["outcomes"].items() if p > 0))
        print("  not-on-pad causes: " + ", ".join(f"{k} {100 * p:.1f}%" for k, p in s["failure_taxonomy"].items())
              + f"; mean wind aloft success {s['wind_aloft_mean_success']:.1f} vs failure {s['wind_aloft_mean_failure']:.1f} m/s")
        print(f"  pad miss median {s['miss_median']:.2f} m, p95 {s['miss_p95']:.2f} m; "
              f"touchdown p95 |vz| {s['vz_p95']}, vh {s['vh_p95']}, tilt {s['tilt_p95']}")
        print(f"  propellant mean {s['prop_mean']:.0f} kg (p95 {s['prop_p95']:.0f}); "
              f"nav pos NEES mean {s['nav_nees_pos_mean']:.2f} (3 = consistent); "
              f"wind err at ignition median {s['wind_err_at_ign_median']:.2f} m/s")

    # ---- figure: landing dispersion + touchdown states ----
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    ax = axs[0]
    for name, col, lab in (("no_wind_ff", ps.SERIES[1], "wind feedforward off"),
                           ("baseline", ps.SERIES[0], "baseline")):
        rows = [r for r in camps[name] if abs(r["x"]) < 40 and abs(r["y"]) < 40]
        ax.plot([r["x"] for r in rows], [r["y"] for r in rows], "o", ms=4.5, color=col, alpha=0.75,
                mec=ps.SURFACE, mew=0.6, label=f"{lab} ({len(rows)}/{len(camps[name])} in view)")
    ax.add_patch(plt.Circle((0, 0), 10, fill=False, ec=ps.INK2, lw=1, ls="--"))
    ax.set_aspect("equal")
    ax.set(xlim=(-20, 20), ylim=(-20, 20), title="Touchdown points (pad radius 10 m)", xlabel="x [m]", ylabel="y [m]")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axs[1]
    soft = [r for r in camps["baseline"] if r["outcome"].startswith("landed") or r["outcome"] == "hard_landing"]
    ax.plot([r["vh"] for r in soft], [-r["vz"] for r in soft], "o", ms=4.5, color=ps.SERIES[0], alpha=0.75,
            mec=ps.SURFACE, mew=0.6)
    ax.axvline(1.5, color=ps.INK2, lw=1, ls="--")
    ax.axhline(2.5, color=ps.INK2, lw=1, ls="--")
    off = sum((-r["vz"] > 20) or (r["vh"] > 16) for r in soft)
    ax.set(xlim=(0, 16), ylim=(0, 20), title="Touchdown velocity (limits dashed)", xlabel="horizontal [m/s]",
           ylabel="vertical [m/s]")
    ax.text(15.5, 19, f"{off} propellant-exhaustion impacts off scale", ha="right", va="top", fontsize=8, color=ps.INK2)
    ax = axs[2]
    b = camps["baseline"]
    ax.hist([r["prop"] / 1000 for r in b], bins=25, color=ps.SERIES[0], alpha=0.85, label="baseline")
    ax.hist([r["prop"] / 1000 for r in camps["no_wind_ff"]], bins=25, color=ps.SERIES[1], alpha=0.55,
            label="wind feedforward off")
    ax.axvline(30, color=ps.INK2, lw=1, ls="--")
    ax.set(title="Landing propellant (30 t loaded, dashed)", xlabel="propellant used [t]", ylabel="runs")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "mc_landing.png")

    # ---- figure: engine-out outcome vs failure time ----
    eo = camps["engine_out"]
    if eo:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        colors = {"landed_pad": ps.SERIES[2], "landed_backup": ps.SERIES[0], "landed_offsite": ps.SERIES[3],
                  "hard_landing": ps.SERIES[1], "crash": ps.SERIES[7], "timeout": ps.MUTED}
        names = ["centre", "outboard +y", "outboard −y"]
        for o in OUTCOMES:
            rows = [r for r in eo if r["outcome"] == o]
            if rows:
                ax.plot([r["eng_fail_time"] for r in rows], [r["eng_fail_index"] + np.random.uniform(-0.15, 0.15) for r in rows],
                        "o", ms=6, color=colors[o], mec=ps.SURFACE, mew=0.8, label=o.replace("_", " "))
        ax.set_yticks([0, 1, 2], names)
        ax.set(title="Single engine-out: outcome vs failure time", xlabel="failure time from glide start [s]  "
               "(glide ≈ 0–120 s, flip/burn ≈ 120–135 s, terminal after)", ylabel="failed engine")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "mc_engine_out.png")
    plt.close("all")


if __name__ == "__main__":
    main()
