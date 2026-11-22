"""Closed-loop software-in-the-loop tests: the C flight software flying the 6-DOF truth model."""
import numpy as np
import pytest

from sim.fsw_bindings import LIB_PATH
from sim.params import FaultParams, Scenario

pytestmark = pytest.mark.skipif(not LIB_PATH.exists(), reason="build fsw first (make fsw)")


def _fly(sc):
    # each test runs in its own process: the C flight software keeps static state
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1) as ex:
        return ex.submit(_run, sc).result()


def _run(sc):
    from sim.fsw_bindings import FlightSoftware
    from sim.simulation import Simulation
    out = Simulation(sc, FlightSoftware(sc), log_every=10).run()
    tl = out["tlm"]
    return dict(outcome=out["outcome"], td=out["touchdown"], modes=np.unique(tl["mode"]).tolist(),
                sens_failed=tl["sens_failed"][-1].tolist(), eng_failed=tl["eng_failed"][-1].tolist(),
                target=int(tl["target_id"][-1]), prop=out["prop_used"])


def test_nominal_lands_on_pad():
    r = _fly(Scenario(seed=3))
    assert r["outcome"] == "landed_pad", r
    assert r["td"]["d_pad"] < 3.0
    assert -r["td"]["vz"] < 2.0 and r["td"]["vh"] < 1.0 and r["td"]["tilt"] < 5.0
    assert {1, 2, 3, 4} <= set(r["modes"])            # glide, flip, burn, terminal all flown


def test_engine_out_in_burn_still_lands_on_pad():
    r = _fly(Scenario(seed=4, faults=FaultParams(engine_fail_time=134.0, engine_fail_index=1)))
    assert r["eng_failed"][1] == 1
    assert r["outcome"] == "landed_pad", r


def test_gnss_step_fault_is_isolated():
    r = _fly(Scenario(seed=5, faults=FaultParams(gnss_step_bias=(100.0, (30.0, -20.0, 0.0)))))
    assert r["sens_failed"][0] == 1
    assert r["outcome"] == "landed_pad", r


def test_double_engine_out_before_ignition_retargets_backup():
    r = _fly(Scenario(seed=6, faults=FaultParams(engine_failures=((60.0, 0), (60.0, 1)))))
    assert r["target"] == 1                            # flight rule: no 1-engine burn over the pad
    assert r["td"]["d_pad"] > 300.0                    # whatever happens, it happens away from the pad
