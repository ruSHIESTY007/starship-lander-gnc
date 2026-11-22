"""
Learned landing-burn guidance: imitation learning of the convex-optimal (G-FOLD) policy
with DAgger, deployed as a fixed-size MLP in the C flight software.

  1. Expert dataset: G-FOLD (free final time) from random burn-start states; every
     discretisation node gives a (state -> optimal thrust acceleration) pair (principle of
     optimality: the tail of an optimal trajectory is optimal from its own start).
  2. Behaviour cloning: 8-64-64-3 tanh MLP, Adam, trained in numpy.
  3. DAgger (Ross et al., 2011): roll the *learned* policy out in the point-mass harness
     (through the C inference code), query the expert at the states the policy actually
     visits, aggregate and retrain -- this removes the compounding-error drift of plain
     behaviour cloning.
  4. Export the flat parameter blob consumed by fsw/src/nn_policy.c.

    .venv/bin/python analysis/learned_guidance.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np

MODEL_DIR = ROOT / "fsw" / "models"
OUT = ROOT / "out"
N_IN, N_H, N_OUT = 8, 64, 3


# ------------------------------------------------------------------ expert
def _features(r, v, a_aero_z, m):
    from gfold import GATE_R
    return np.concatenate([np.asarray(r) - GATE_R, v, [a_aero_z, m * 1e-5]])


def expert_trajectory(args):
    """Solve G-FOLD from a state; return (X, Y) at all nodes, or None."""
    r0, v0, m0, aero = args
    import warnings
    warnings.filterwarnings("ignore")
    from gfold import solve
    res = solve(r0, v0, m0, t_lo=1.0, t_hi=30.0, a_aero_z=aero)
    if res is None:
        return None
    X = np.array([_features(res.r[k], res.v[k], aero, res.m[k]) for k in range(len(res.t) - 1)])
    return X, res.u[:-1].copy(), res.fuel


def expert_action(args):
    """Optimal first action from a single state (DAgger query)."""
    r0, v0, m0, aero = args
    import warnings
    warnings.filterwarnings("ignore")
    from gfold import solve
    res = solve(r0, v0, m0, t_lo=0.5, t_hi=30.0, a_aero_z=aero)
    if res is None:
        return None
    return _features(r0, v0, aero, m0), res.u[0].copy()


# ------------------------------------------------------------------ MLP (numpy)
class MLP:
    def __init__(self, rng):
        def glorot(o, i):
            return rng.normal(0, np.sqrt(2.0 / (i + o)), (o, i))
        self.W1, self.b1 = glorot(N_H, N_IN), np.zeros(N_H)
        self.W2, self.b2 = glorot(N_H, N_H), np.zeros(N_H)
        self.W3, self.b3 = glorot(N_OUT, N_H), np.zeros(N_OUT)
        self.xm = self.xs = self.ym = self.ys = None

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, Xn):
        h1 = np.tanh(Xn @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        return h2 @ self.W3.T + self.b3, (Xn, h1, h2)

    def fit(self, X, Y, epochs=300, batch=256, lr=2e-3, rng=None, log=True):
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-6
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-6
        Xn, Yn = (X - self.xm) / self.xs, (Y - self.ym) / self.ys
        P = self.params()
        mom = [np.zeros_like(p) for p in P]
        vel = [np.zeros_like(p) for p in P]
        b1, b2, step = 0.9, 0.999, 0
        n = len(X)
        for ep in range(epochs):
            idx = rng.permutation(n)
            lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))      # cosine decay
            for s in range(0, n, batch):
                j = idx[s:s + batch]
                out, (x0, h1, h2) = self.forward(Xn[j])
                d3 = 2.0 * (out - Yn[j]) / len(j)
                g3w, g3b = d3.T @ h2, d3.sum(0)
                d2 = (d3 @ self.W3) * (1 - h2 ** 2)
                g2w, g2b = d2.T @ h1, d2.sum(0)
                d1 = (d2 @ self.W2) * (1 - h1 ** 2)
                g1w, g1b = d1.T @ x0, d1.sum(0)
                grads = [g1w, g1b, g2w, g2b, g3w, g3b]
                step += 1
                for p, gr, mo, ve in zip(P, grads, mom, vel):
                    mo *= b1; mo += (1 - b1) * gr
                    ve *= b2; ve += (1 - b2) * gr * gr
                    p -= lr_ep * (mo / (1 - b1 ** step)) / (np.sqrt(ve / (1 - b2 ** step)) + 1e-8)
            if log and (ep % 100 == 0 or ep == epochs - 1):
                pred, _ = self.forward(Xn)
                print(f"    epoch {ep:4d}  normalised MSE {np.mean((pred - Yn) ** 2):.4f}")

    def predict(self, X):
        out, _ = self.forward((X - self.xm) / self.xs)
        return out * self.ys + self.ym

    def blob(self):
        """Flat float32 parameter vector in the layout of fsw/src/nn_policy.c."""
        parts = [self.W1.ravel(), self.b1, self.W2.ravel(), self.b2, self.W3.ravel(), self.b3,
                 self.xm, self.xs, self.ym, self.ys]
        return np.concatenate(parts).astype(np.float32)


# ------------------------------------------------------------------ DAgger rollouts
def _rollout_visits(args):
    blob, r0, v0, m0, aero, n_keep, seed = args
    import cguid
    from pointmass import rollout
    cguid.load_nn(blob)
    res = rollout("nn", r0, v0, m0, aero, record=True)
    st = res["states"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(st), size=min(n_keep, len(st)), replace=False) if st else []
    return [(st[k][1], st[k][2], st[k][3], aero) for k in pick], res


def main(n_expert=500, dagger_iters=3, n_roll=160, seed=0):
    from pointmass import sample_states
    rng = np.random.default_rng(seed)
    OUT.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r, v, m, aero = sample_states(rng, n_expert)
    with ProcessPoolExecutor(14) as ex:
        trajs = [x for x in ex.map(expert_trajectory, zip(r, v, m, aero), chunksize=4) if x is not None]
    X = np.concatenate([t[0] for t in trajs])
    Y = np.concatenate([t[1] for t in trajs])
    print(f"expert: {len(trajs)}/{n_expert} G-FOLD trajectories, {len(X)} samples ({time.time() - t0:.0f} s)")
    net = MLP(rng)
    print("behaviour cloning:")
    net.fit(X, Y, rng=rng)
    history = []
    for it in range(dagger_iters):
        r, v, m, aero = sample_states(rng, n_roll)
        blob = net.blob()
        jobs = [(blob, r[i], v[i], m[i], aero[i], 10, seed * 1000 + it * 100 + i) for i in range(n_roll)]
        with ProcessPoolExecutor(14) as ex:
            outs = list(ex.map(_rollout_visits, jobs))
        visits = [s for o in outs for s in o[0]]
        succ = np.mean([o[1]["dr_h"] < 2.0 and o[1]["dv"] < 1.0 for o in outs])
        history.append(dict(iter=it, rollout_success=float(succ), n_visits=len(visits)))
        print(f"DAgger iter {it}: policy rollout success {succ:.2%}; querying expert at {len(visits)} states")
        with ProcessPoolExecutor(14) as ex:
            lab = [x for x in ex.map(expert_action, visits, chunksize=8) if x is not None]
        X = np.concatenate([X, np.array([a for a, _ in lab])])
        Y = np.concatenate([Y, np.array([b for _, b in lab])])
        net = MLP(rng)
        net.fit(X, Y, rng=rng, log=False)
        print(f"  aggregated dataset {len(X)} samples")
    blob = net.blob()
    blob.tofile(MODEL_DIR / "nn_policy.f32")
    json.dump(dict(history=history, n_samples=int(len(X)), n_params=int(blob.size),
                   wall_s=time.time() - t0), open(OUT / "learned_guidance.json", "w"), indent=2)
    print(f"saved {blob.size} parameters to {MODEL_DIR / 'nn_policy.f32'} ({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()
