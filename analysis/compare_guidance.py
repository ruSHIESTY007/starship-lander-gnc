"""
Landing-burn guidance comparison on held-out burn-start states (point-mass harness):
  * G-FOLD (offline convex optimum; lower bound on propellant)
  * ZEM/ZEV (onboard), swept over the time weight Gamma
  * learned MLP policy (pure) and learned -> ZEM/ZEV hybrid with runtime monitor
Writes out/guidance_comparison.json and docs/figures/guidance_*.png.
"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np

GAMMAS = [1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 40.0]
BLOB = ROOT / "fsw" / "models" / "nn_policy.f32"


def evaluate(args):
    warnings.filterwarnings("ignore")
    import cguid
    from gfold import solve
    from pointmass import rollout
    r, v, m, a = args
    cguid.load_nn(np.fromfile(BLOB, dtype=np.float32))
    g = solve(r, v, m, t_lo=1.0, t_hi=30.0, a_aero_z=a)
    if g is None:
        return None
    res = {"gfold": dict(fuel=g.fuel, tf=g.tf)}
    for gam in GAMMAS:
        res[f"zemzev_{gam:g}"] = rollout("zemzev", r, v, m, a, gamma=gam)
    res["nn"] = rollout("nn", r, v, m, a)
    res["hybrid"] = rollout("hybrid", r, v, m, a, gamma=6.0)
    return res


def ok(x):
    return x["dr_h"] < 1.0 and x["dv"] < 1.0 and abs(x.get("dz", 0)) < 1.0


def main(n=200, seed=2026):
    from pointmass import sample_states
    rng = np.random.default_rng(seed)
    r, v, m, a = sample_states(rng, n)
    with ProcessPoolExecutor(14) as ex:
        rows = [x for x in ex.map(evaluate, zip(r, v, m, a), chunksize=4) if x is not None]
    laws = [f"zemzev_{g:g}" for g in GAMMAS] + ["nn", "hybrid"]
    summary = {"n_states": len(rows), "gfold_fuel_mean": float(np.mean([x["gfold"]["fuel"] for x in rows]))}
    for law in laws:
        gap = np.array([x[law]["fuel"] / x["gfold"]["fuel"] - 1 for x in rows]) * 100
        summary[law] = dict(fuel_gap_pct_mean=float(gap.mean()), fuel_gap_pct_p95=float(np.percentile(gap, 95)),
                            gate_success=float(np.mean([ok(x[law]) for x in rows])),
                            dv_mean=float(np.mean([x[law]["dv"] for x in rows])),
                            drh_mean=float(np.mean([x[law]["dr_h"] for x in rows])))
    (ROOT / "out").mkdir(exist_ok=True)
    json.dump(summary, open(ROOT / "out" / "guidance_comparison.json", "w"), indent=2)
    for law in laws:
        s = summary[law]
        print(f"{law:12s} fuel +{s['fuel_gap_pct_mean']:5.1f}% (p95 +{s['fuel_gap_pct_p95']:5.1f}%)  "
              f"gate success {s['gate_success']:6.1%}  mean |dv| {s['dv_mean']:.2f} m/s  mean |dr_h| {s['drh_mean']:.2f} m")
    plots(summary, r[0], v[0], m[0], a[0])


def plots(summary, r0, v0, m0, a0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotstyle as ps
    import cguid
    from gfold import solve
    from pointmass import rollout
    ps.apply()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    gaps = [summary[f"zemzev_{g:g}"]["fuel_gap_pct_mean"] for g in GAMMAS]
    ax.plot(GAMMAS, gaps, "-o", color=ps.SERIES[0], ms=6, mec=ps.SURFACE, mew=1.5, label="ZEM/ZEV (onboard)")
    ax.axhline(summary["hybrid"]["fuel_gap_pct_mean"], color=ps.SERIES[1], lw=1.8, label="learned → ZEM/ZEV hybrid")
    ax.axhline(0, color=ps.INK2, lw=1.0, ls="--", label="G-FOLD optimum")
    ax.set_xscale("log")
    ax.set(title="Propellant above the convex optimum (200 held-out states)", xlabel="ZEM/ZEV time weight Γ",
           ylabel="mean extra propellant [%]")
    ax.legend(loc="upper right")
    ax = axs[1]
    cguid.load_nn(np.fromfile(BLOB, dtype=np.float32))
    g = solve(r0, v0, m0, t_lo=1.0, t_hi=30.0, a_aero_z=a0)
    ax.plot(g.t, np.linalg.norm(g.u, axis=1) * g.m / 1e6, color=ps.INK2, lw=1.4, ls="--", label="G-FOLD (bang-bang)")
    for law, col, lab in (("zemzev", ps.SERIES[0], "ZEM/ZEV Γ=6"), ("hybrid", ps.SERIES[1], "learned → ZEM/ZEV")):
        res = rollout(law, r0, v0, m0, a0, gamma=6.0, record=True)
        st = res["states"]
        tt = np.array([s[0] for s in st])
        # thrust from finite-difference of mass (exact for the integrator used)
        mm = np.array([s[3] for s in st])
        from gfold import ALPHA
        from pointmass import DT
        T = -np.gradient(mm, DT) / ALPHA / 1e6
        ax.plot(tt, T, color=col, label=lab)
    ax.set(title="Thrust profile, one burn", xlabel="time from burn start [s]", ylabel="thrust [MN]")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "guidance_comparison.png")


if __name__ == "__main__":
    main()
