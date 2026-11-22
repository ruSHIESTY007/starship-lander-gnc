"""Fill the README result tables from out/*.json (keeps the README honest and reproducible)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(x):
    return f"{100 * x:.1f} %"


def main(bench_text=""):
    mc = json.load(open(ROOT / "out" / "mc_summary.json"))
    gc = json.load(open(ROOT / "out" / "guidance_comparison.json"))
    b, nw, eo, nn = mc["baseline"], mc["no_wind_ff"], mc["engine_out"], mc["neural"]
    glance = f"""| Metric (6-DOF SIL Monte Carlo, dispersed) | Result |
|---|---|
| Lands on the 10 m pad (baseline, n={b['n']}) | **{pct(b['outcomes']['landed_pad'])}** |
| Pad miss distance when on the pad: median / p95 | **{b['miss_median']:.2f} m / {b['miss_p95']:.2f} m** |
| Touchdown p95: vertical / horizontal speed / tilt | {b['vz_p95']:.2f} m/s / {b['vh_p95']:.2f} m/s / {b['tilt_p95']:.1f}° |
| Random single engine-out anywhere in flight (n={eo['n']}) → on pad | **{pct(eo['outcomes']['landed_pad'])}** (same as no failure) |
| Wind feedforward on vs off: p95 pad miss (pad-landing rate) | {b['miss_p95']:.2f} m vs {nw['miss_p95']:.2f} m ({pct(b['outcomes']['landed_pad'])} vs {pct(nw['outcomes']['landed_pad'])}) |
| Navigation consistency: mean position NEES (3 = ideal) | {b['nav_nees_pos_mean']:.2f} |
| Landing-burn propellant vs convex optimum (point-mass, 200 states) | ZEM/ZEV +{gc['zemzev_6']['fuel_gap_pct_mean']:.1f} %, learned hybrid +{gc['hybrid']['fuel_gap_pct_mean']:.1f} % |
| Dominant failure: propellant exhaustion after a wind-driven glide miss | {pct(b['failure_taxonomy']['propellant_exhausted'])} of runs |

The headline limitation is stated up front: when strong winds aloft push the glide off target the
landing burn and terminal phase have to recover hundreds of metres and the 30 t load runs out
(see *Known limitations*)."""
    rows = []
    names = {"baseline": "baseline", "no_wind_ff": "wind feedforward off", "engine_out": "random engine-out",
             "neural": "learned → ZEM/ZEV burn"}
    for k, s in mc.items():
        if not s:
            continue
        o, ft = s["outcomes"], s["failure_taxonomy"]
        rows.append(f"| {names[k]} | {s['n']} | {pct(o['landed_pad'])} | {pct(o.get('landed_offsite', 0) + o.get('landed_backup', 0))} | "
                    f"{s['miss_median']:.2f} / {s['miss_p95']:.2f} m | {s['prop_mean'] / 1000:.1f} t | "
                    f"{pct(ft['propellant_exhausted'])} | {pct(ft['far_from_pad_gt100m'])} | {s['nav_nees_pos_mean']:.2f} |")
    detail = ("| Campaign | n | on pad | soft, off pad | miss median / p95 | propellant | prop. exhausted | >100 m off | nav NEES |\n"
              "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) +
              f"\n\nMean wind aloft in runs that reached the pad: {b['wind_aloft_mean_success']:.1f} m/s; in runs that did not: "
              f"{b['wind_aloft_mean_failure']:.1f} m/s. The wind estimate at ignition has a median error of "
              f"{b['wind_err_at_ign_median']:.1f} m/s (truth includes turbulence), which is good enough for the terminal "
              "feedforward (it halves the p95 miss) but does not rescue a glide that is already out of reach; the upwind "
              "aim offset it adds even costs a few percent of pad landings in strong-wind runs, so the aim-point use "
              "should be gated on estimate confidence. "
              "The learned burn policy, trained in the aero-free point-mass harness, transfers poorly to full 6-DOF "
              "flight: the hybrid monitor keeps landings precise when it lands, but the network's long low-thrust arcs "
              "cost time and propellant under real aero and attitude dynamics, so it is a research result here, not the "
              "flight default.")
    gt = ["| Law | extra propellant vs G-FOLD (mean / p95) | gate success (<1 m, <1 m/s) | mean gate |Δv| |", "|---|---|---|---|"]
    for k, lab in [("zemzev_1", "ZEM/ZEV, Γ = 1"), ("zemzev_6", "ZEM/ZEV, Γ = 6 (flight)"), ("zemzev_20", "ZEM/ZEV, Γ = 20"),
                   ("zemzev_40", "ZEM/ZEV, Γ = 40"), ("nn", "learned MLP alone"), ("hybrid", "learned → ZEM/ZEV hybrid")]:
        g = gc[k]
        gt.append(f"| {lab} | +{g['fuel_gap_pct_mean']:.1f} % / +{g['fuel_gap_pct_p95']:.1f} % | {pct(g['gate_success'])} | {g['dv_mean']:.2f} m/s |")
    readme = (ROOT / "README.md").read_text()
    readme = readme.replace("<!-- MC_TABLE -->", glance)
    readme = readme.replace("<!-- MC_DETAIL -->", detail)
    readme = readme.replace("<!-- GUIDANCE_TABLE -->", "\n".join(gt))
    if bench_text:
        readme = readme.replace("<!-- EMBEDDED_TABLE -->", bench_text)
    (ROOT / "README.md").write_text(readme)
    print("README filled")


if __name__ == "__main__":
    import sys
    main(open(sys.argv[1]).read() if len(sys.argv) > 1 else "")
