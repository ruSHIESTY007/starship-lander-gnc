"""
6-DOF Monte Carlo of the full flight software (SIL), with dispersed vehicle, environment
and sensors. Campaigns:
  baseline   -- nominal FSW (wind estimation + feedforward, ZEM/ZEV burn)
  no_wind_ff -- same seeds, wind estimate not used in guidance feedforward / aim point
  engine_out -- one random engine fails at a random time between glide start and landing
  neural     -- learned -> ZEM/ZEV hybrid burn guidance
    .venv/bin/python analysis/monte_carlo.py [n_baseline]
Results: out/mc_<campaign>.json
"""
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def dispersed_scenario(seed, campaign):
    from sim.params import EnvParams, FaultParams, Scenario, VehicleParams
    rng = np.random.default_rng(10_000 + seed)
    nom = VehicleParams()
    veh = replace(
        nom,
        m_prop=nom.m_prop * (1 + rng.normal(0, 0.02)),
        inertia=tuple(np.asarray(nom.inertia) * (1 + rng.normal(0, 0.05, 3))),
        thrust_scale=tuple(1 + rng.normal(0, 0.015, 3)),
        isp_scale=1 + rng.normal(0, 0.01),
        thrust_misalign=tuple(map(tuple, rng.normal(0, 0.002, (3, 2)))),
        com_offset=(rng.normal(0, 0.3), rng.normal(0, 0.05), rng.normal(0, 0.05)),
        cdc_scale=1 + rng.normal(0, 0.08),
        ca_nose_first=nom.ca_nose_first * (1 + rng.normal(0, 0.10)),
        ca_tail_first=nom.ca_tail_first * (1 + rng.normal(0, 0.10)),
        flap_cf=nom.flap_cf * (1 + rng.normal(0, 0.10)),
    )
    ws, wa = rng.uniform(0, 12), rng.uniform(5, 30)
    ds, da = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    env = EnvParams(rho_scale=1 + rng.normal(0, 0.03), temp_offset=rng.normal(0, 5),
                    wind_surface=(ws * math.cos(ds), ws * math.sin(ds)),
                    wind_aloft=(wa * math.cos(da), wa * math.sin(da)),
                    turb_sigma=rng.uniform(0.5, 3.0))
    faults = FaultParams()
    if campaign == "engine_out":
        faults = FaultParams(engine_fail_time=float(rng.uniform(0, 150)), engine_fail_index=int(rng.integers(0, 3)))
    # glide-start (entry handover) box; the glide's reach is limited -- see README
    r0 = (-3200 + rng.normal(0, 100), 350 + rng.normal(0, 100), 10000 + rng.normal(0, 100))
    v0 = (75 + rng.normal(0, 3), -4 + rng.normal(0, 3), -115 + rng.normal(0, 3))
    return Scenario(r0=r0, v0=v0, yaw0_deg=rng.normal(0, 5), seed=seed, veh=veh, env=env, faults=faults)


def run(args):
    seed, campaign = args
    from sim.fsw_bindings import GUID_NEURAL, GUID_ZEMZEV, FlightSoftware
    from sim.simulation import Simulation
    sc = dispersed_scenario(seed, campaign)
    kw = {}
    if campaign == "no_wind_ff":
        kw["use_wind_ff"] = False
    if campaign == "neural":
        kw["guidance_law"] = GUID_NEURAL
        kw["nn_params"] = np.fromfile(ROOT / "fsw" / "models" / "nn_policy.f32", dtype=np.float32)
    t0 = time.time()
    out = Simulation(sc, FlightSoftware(sc, **kw), log_every=10).run()
    tl = out["tlm"]
    td = out["touchdown"] or {}
    fr = out["final_state"]
    modes = tl["mode"]
    k_ign = int(np.argmax(modes >= 2)) if (modes >= 2).any() else len(modes) - 1
    # nav consistency at the end: normalised position error per axis
    nerr = (tl["nav_p"][-1] - out["r"][-1]) / np.maximum(tl["nav_sig"][-1][:3], 1e-6)
    werr = float(np.linalg.norm(tl["wind_est"][k_ign] - out["wind"][k_ign][:2]))
    return dict(seed=seed, campaign=campaign, outcome=out["outcome"],
                d_pad=float(td.get("d_pad", np.linalg.norm(fr[:2]))),
                d_backup=float(td.get("d_backup", np.linalg.norm(fr[:2] - np.array(sc.backup[:2])))),
                x=float(fr[0]), y=float(fr[1]),
                vz=float(td.get("vz", fr[5])), vh=float(td.get("vh", np.linalg.norm(fr[3:5]))),
                tilt=float(td.get("tilt", np.nan)), prop=float(out["prop_used"]),
                target=int(tl["target_id"][-1]), divert_priority=int(tl["divert_priority"][-1]),
                eng_fail_time=sc.faults.engine_fail_time, eng_fail_index=sc.faults.engine_fail_index,
                nav_nerr=nerr.tolist(), wind_err_at_ign=werr,
                wind_surface=float(np.hypot(*sc.env.wind_surface)), wind_aloft=float(np.hypot(*sc.env.wind_aloft)),
                t_flight=float(out["t"][-1]), wall=time.time() - t0)


def main(n_base=200, n_eo=120, n_nn=80):
    camps = [("baseline", n_base), ("no_wind_ff", n_base), ("engine_out", n_eo), ("neural", n_nn)]
    (ROOT / "out").mkdir(exist_ok=True)
    for name, n in camps:
        t0 = time.time()
        with ProcessPoolExecutor(14) as ex:
            rows = list(ex.map(run, [(s, name) for s in range(n)], chunksize=2))
        json.dump(rows, open(ROOT / "out" / f"mc_{name}.json", "w"))
        oc = {}
        for r in rows:
            oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
        print(f"{name:11s} n={n} ({time.time() - t0:.0f} s): {oc}", flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
