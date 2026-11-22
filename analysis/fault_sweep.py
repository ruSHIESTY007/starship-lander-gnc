"""Run the nominal scenario under a set of injected faults (development aid)."""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

CASES = {
    "nominal": {},
    "eng0_out_glide": dict(engine_fail_time=60.0, engine_fail_index=0),
    "eng1_out_glide": dict(engine_fail_time=60.0, engine_fail_index=1),
    "eng1_out_flip": dict(engine_fail_time=128.5, engine_fail_index=1),
    "eng0_out_burn": dict(engine_fail_time=134.0, engine_fail_index=0),
    "eng1_out_burn": dict(engine_fail_time=134.0, engine_fail_index=1),
    "eng0_out_term": dict(engine_fail_time=145.0, engine_fail_index=0),
    "eng01_out_glide": dict(engine_failures=((60.0, 0), (60.0, 1))),
    "eng0_out_burn_eng1_term": dict(engine_failures=((134.0, 0), (141.0, 1))),
    "eng01_out_burn": dict(engine_failures=((134.0, 0), (134.0, 1))),
    "gnss_dropout": dict(gnss_dropout=(100.0, 160.0)),
    "gnss_step": dict(gnss_step_bias=(110.0, (30.0, -20.0, 0.0))),
    "radar_stuck": dict(radar_stuck_time=125.0),
}

def run(name):
    from sim.params import Scenario, FaultParams
    from sim.simulation import Simulation
    from sim.fsw_bindings import FlightSoftware
    sc = Scenario(faults=FaultParams(**CASES[name]))
    out = Simulation(sc, FlightSoftware(sc)).run()
    td = out["touchdown"] or {}
    tl = out["tlm"]
    return (name, out["outcome"], td.get("d_pad", np.nan), td.get("d_backup", np.nan), td.get("vz", np.nan),
            td.get("vh", np.nan), td.get("tilt", np.nan), out["prop_used"], int(tl["target_id"][-1]),
            tl["sens_failed"][-1].tolist(), tl["eng_failed"][-1].tolist())

if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    with ProcessPoolExecutor(max_workers=min(len(names), 12)) as ex:
        for r in ex.map(run, names):
            print(f"{r[0]:16s} {r[1]:15s} d_pad={r[2]:7.1f} d_bak={r[3]:7.1f} vz={r[4]:6.2f} vh={r[5]:5.2f} "
                  f"tilt={r[6]:5.1f} prop={r[7]:6.0f} tgt={r[8]} sens_fail={r[9]} eng_fail={r[10]}")
