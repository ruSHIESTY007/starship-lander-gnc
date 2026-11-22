"""Quick closed-loop run with a mode-transition trace (development aid)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sim.params import Scenario
from sim.simulation import Simulation
from sim.fsw_bindings import FlightSoftware, MODE_NAMES
from sim import quat

def main(**kw):
    sc = Scenario(**kw)
    t0 = time.time()
    out = Simulation(sc, FlightSoftware(sc)).run()
    tl = out["tlm"]; t = out["t"]
    print(f"wall {time.time()-t0:.1f}s  outcome={out['outcome']}  prop={out['prop_used']:.0f} kg")
    modes = tl["mode"]
    for k in np.r_[0, np.nonzero(np.diff(modes))[0] + 1]:
        r, v = out["r"][k], out["v"][k]
        print(f"  t={t[k]:6.2f} {MODE_NAMES[int(modes[k])]:9s} r=({r[0]:7.1f},{r[1]:6.1f},{r[2]:7.1f}) "
              f"v=({v[0]:6.1f},{v[1]:6.1f},{v[2]:6.1f}) tilt={quat.tilt_deg(out['q'][k]):5.1f} "
              f"m={out['m'][k]:.0f} h_ign={tl['h_ign'][k]:.0f}")
    for k in range(0, len(t), max(len(t)//25, 1)):
        r, v = out["r"][k], out["v"][k]
        e = np.degrees(np.linalg.norm(tl["att_err"][k]))
        print(f"  t={t[k]:6.1f} {MODE_NAMES[int(modes[k])]:8s} z={r[2]:7.1f} x={r[0]:7.1f} y={r[1]:6.1f} "
              f"vx={v[0]:6.1f} vz={v[2]:6.1f} alpha={out['alpha'][k]:6.1f} atterr={e:5.1f} "
              f"T={out['T'][k].sum()/1e6:4.2f}MN flaps={np.round(out['flaps'][k],2)} sat={tl['alloc_saturated'][k]}")
    print("touchdown:", out["touchdown"])
    return out

if __name__ == "__main__":
    main()
