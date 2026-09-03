"""Sharpness probe for the tiny-circuit solutions.

Retrains a few reference configs with `save_params=True` (into
runs/iter_params, so the existing runs/iter results are untouched), then
measures each final solution's train-pool / held-out BCE under init-relative
Gaussian weight noise of increasing size. A memorizing solution that relies
on fine cancellations should degrade fast; a solution that implements the
circuit should be flat.

  uv run python probe.py            # train (idempotent) + print table
"""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from iter_grid import BASE, SEEDS
from mlp import MLPConfig, forward, init_stds
from random_circuit import make_jax_evaluator, sample_circuit
from train import RunConfig, run

OUT_DIR = "runs/iter_params"
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
N_DRAWS = 64

PROBES = {
    "clean": [dict(model_seed=s) for s in SEEDS],
    "wnr0.5": [dict(weight_noise=0.5, model_seed=s) for s in SEEDS],
    "wnr1": [dict(weight_noise=1.0, model_seed=s) for s in SEEDS],
    "is0.2": [dict(init_scale=0.2, model_seed=s) for s in SEEDS],
    "is0.5_wnr0.5": [dict(init_scale=0.5, weight_noise=0.5, model_seed=s) for s in SEEDS],
    "tf0.3_wnr1": [dict(train_frac=0.3, weight_noise=1.0, model_seed=s) for s in SEEDS],
    "tf0.3_clean": [dict(train_frac=0.3, model_seed=s) for s in SEEDS],
}


def cfg(**kw) -> RunConfig:
    return RunConfig(**{**BASE, "out_dir": OUT_DIR, "save_params": True, **kw})


def task_arrays(c: RunConfig):
    circuit = sample_circuit(np.random.default_rng(c.circuit_seed), c.n_wires, c.circ_depth)
    ev = make_jax_evaluator(circuit)
    n_all = 2 ** c.n_wires
    all_x = ((np.arange(n_all)[:, None] >> np.arange(c.n_wires)[::-1]) & 1).astype(np.uint8)
    n_pool = int(round(c.train_frac * n_all))
    pool = np.sort(np.random.default_rng(c.pool_seed).choice(n_all, n_pool, replace=False))
    in_pool = np.zeros(n_all, dtype=bool)
    in_pool[pool] = True
    x = jnp.asarray(all_x)
    y = jnp.asarray(np.asarray(ev(x)), dtype=jnp.float32)
    return x, y, jnp.asarray(in_pool)


def sharpness(params, c: RunConfig, x, y, in_pool, key):
    """-> dict sigma -> (train BCE, held-out BCE, held-out acc, train flip rate),
    averaged over N_DRAWS noise draws (sigma=0: the clean values)."""
    mcfg = MLPConfig(n_inputs=c.n_wires, n_outputs=c.n_wires, width=c.width,
                     depth=c.mlp_depth, hidden_ratio=c.hidden_ratio)
    stds = jax.tree_util.tree_leaves(init_stds(mcfg))
    leaves, tdef = jax.tree_util.tree_flatten(params)
    w = c.output_wires[0]
    clean_pred = forward(params, x)[:, w] > 0

    @jax.jit
    def metrics(sig, k):
        keys = jax.random.split(k, len(leaves))
        p = jax.tree_util.tree_unflatten(tdef, [
            l + sig * s * jax.random.normal(kk, l.shape, l.dtype)
            for l, s, kk in zip(leaves, stds, keys)])
        logit = forward(p, x)[:, w]
        bce = jax.nn.softplus(logit) - logit * y[:, w]
        acc = ((logit > 0) == (y[:, w] > 0.5)).astype(jnp.float32)
        flip = ((logit > 0) != clean_pred).astype(jnp.float32)
        tr, ho = in_pool.astype(jnp.float32), (~in_pool).astype(jnp.float32)
        return jnp.stack([bce @ tr / tr.sum(), bce @ ho / ho.sum(),
                          acc @ ho / ho.sum(), flip @ tr / tr.sum()])

    out = {}
    for sig in SIGMAS:
        ks = jax.random.split(jax.random.fold_in(key, int(sig * 1000)), N_DRAWS)
        out[sig] = np.mean([np.asarray(metrics(sig, k)) for k in ks], axis=0)
    return out


def main():
    rows = []
    for label, overrides in PROBES.items():
        for o in overrides:
            c = cfg(**o)
            run(c, quiet=True)
            with open(c.params_path, "rb") as f:
                params = pickle.load(f)
            x, y, in_pool = task_arrays(c)
            sh = sharpness(params, c, x, y, in_pool, jax.random.key(123))
            pn = float(np.sqrt(sum(float(jnp.vdot(p, p))
                                   for p in jax.tree_util.tree_leaves(params))))
            rows.append((label, c.model_seed, pn, sh))
            print(f"[probe] {label} ms{c.model_seed} done", flush=True)

    print("\ntrain-pool BCE under init-relative weight noise sigma (mean of "
          f"{N_DRAWS} draws)")
    print(f"{'config':<14s}{'ms':>3s}{'|w|':>7s}" + "".join(f"{s:>8.1f}" for s in SIGMAS))
    for label, ms, pn, sh in rows:
        print(f"{label:<14s}{ms:>3d}{pn:>7.1f}" + "".join(f"{sh[s][0]:>8.3f}" for s in SIGMAS))
    print("\nheld-out acc under noise")
    print(f"{'config':<14s}{'ms':>3s}{'|w|':>7s}" + "".join(f"{s:>8.1f}" for s in SIGMAS))
    for label, ms, pn, sh in rows:
        print(f"{label:<14s}{ms:>3d}{pn:>7.1f}" + "".join(f"{sh[s][2]:>8.3f}" for s in SIGMAS))
    print("\ntrain-pool prediction flip rate under noise")
    print(f"{'config':<14s}{'ms':>3s}{'|w|':>7s}" + "".join(f"{s:>8.1f}" for s in SIGMAS))
    for label, ms, pn, sh in rows:
        print(f"{label:<14s}{ms:>3d}{pn:>7.1f}" + "".join(f"{sh[s][3]:>8.3f}" for s in SIGMAS))


if __name__ == "__main__":
    main()
