"""One training run of the scaling sweep: online training on a seeded circuit.

Each run is identified by its config-derived name and produces
`<out_dir>/<name>.npz` holding the full eval time-series (each eval checkpoint
is an (N, D) datapoint for scaling fits). Runs checkpoint every
`checkpoint_every` steps and resume exactly (the batch at step t is derived by
folding t into the data seed, so an interrupted run continues bit-for-bit);
completed runs are skipped.

CLI: uv run python train.py --width 256 --lr 1e-3 --steps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from mlp import MLPConfig, forward, init_params, per_output_bce
from random_circuit import make_jax_evaluator, sample_circuit


@dataclass(frozen=True)
class RunConfig:
    # model
    width: int = 256
    mlp_depth: int = 4
    hidden_ratio: int = 4
    # training
    batch: int = 256
    steps: int = 50_000
    lr: float = 1e-3
    warmup: int = 500
    schedule: str = "constant"  # "constant" (honest mid-run L(N,D) points) or "cosine"
    # circuit / seeds
    n_wires: int = 256
    circ_depth: int = 32
    circuit_seed: int = 0
    model_seed: int = 0
    data_seed: int | None = None  # None -> model_seed (vary init AND data order)
    eval_seed: int = 1            # keep fixed across the whole sweep
    # loss restriction: train only on these output wires (None = all 256).
    # Eval still records every output, so incidental learning is visible.
    output_wires: tuple | None = None
    # eval / io
    eval_every: int = 500
    eval_n: int = 4000
    checkpoint_every: int = 2500
    out_dir: str = "runs"

    @property
    def name(self) -> str:
        s = f"w{self.width}_d{self.mlp_depth}_b{self.batch}_lr{self.lr:g}_s{self.steps}"
        if self.schedule != "constant":
            s += f"_{self.schedule}"
        if self.output_wires is not None:
            s += "_ow" + "-".join(map(str, self.output_wires))
        return s + f"_cs{self.circuit_seed}_ms{self.model_seed}"

    @property
    def npz_path(self) -> Path:
        return Path(self.out_dir) / f"{self.name}.npz"

    @property
    def ckpt_path(self) -> Path:
        return Path(self.out_dir) / f"{self.name}.ckpt.pkl"


def _make_schedule(cfg: RunConfig):
    if cfg.schedule == "constant":
        return optax.join_schedules(
            [optax.linear_schedule(0.0, cfg.lr, cfg.warmup),
             optax.constant_schedule(cfg.lr)],
            [cfg.warmup],
        )
    if cfg.schedule == "cosine":
        return optax.warmup_cosine_decay_schedule(
            0.0, cfg.lr, cfg.warmup, cfg.steps, end_value=0.1 * cfg.lr
        )
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def _save_ckpt(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(state), f)
    os.replace(tmp, path)


def run(cfg: RunConfig, stop_after: int | None = None, quiet: bool = False):
    """Execute (or resume) a run. Returns the npz path, or None if stopped
    early by `stop_after` (a step budget for this invocation; mainly for
    tests) — in that case a checkpoint is left behind and a later call
    resumes it."""
    if cfg.npz_path.exists():
        if not quiet:
            print(f"[skip] {cfg.name} (done)")
        return cfg.npz_path
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    circuit = sample_circuit(
        np.random.default_rng(cfg.circuit_seed), cfg.n_wires, cfg.circ_depth
    )
    circ_eval = make_jax_evaluator(circuit)
    mlp_cfg = MLPConfig(
        n_inputs=cfg.n_wires, n_outputs=cfg.n_wires,
        width=cfg.width, depth=cfg.mlp_depth, hidden_ratio=cfg.hidden_ratio,
    )
    opt = optax.adam(_make_schedule(cfg))
    wire_mask = None if cfg.output_wires is None else jnp.asarray(cfg.output_wires)
    data_key = jax.random.key(cfg.model_seed if cfg.data_seed is None else cfg.data_seed)

    if cfg.ckpt_path.exists():
        with open(cfg.ckpt_path, "rb") as f:
            st = pickle.load(f)
        if not quiet:
            print(f"[resume] {cfg.name} at step {st['step']}")
    else:
        params = init_params(jax.random.key(cfg.model_seed), mlp_cfg)
        st = {
            "step": 0,
            "params": params,
            "opt_state": opt.init(params),
            "train_loss": np.zeros(cfg.steps, dtype=np.float32),
            "eval_steps": [],
            "per_out_loss": [],
            "per_out_acc": [],
            "wall_time": 0.0,
        }

    @jax.jit
    def train_step(params, opt_state, step):
        key = jax.random.fold_in(data_key, step)
        x = jax.random.bernoulli(key, shape=(cfg.batch, cfg.n_wires)).astype(jnp.uint8)
        y = circ_eval(x).astype(jnp.float32)
        def loss_fn(p):
            pol = per_output_bce(forward(p, x), y)
            return jnp.mean(pol if wire_mask is None else pol[wire_mask])

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    eval_x = jax.random.bernoulli(
        jax.random.key(cfg.eval_seed), shape=(cfg.eval_n, cfg.n_wires)
    ).astype(jnp.uint8)
    eval_y = circ_eval(eval_x).astype(jnp.float32)

    @jax.jit
    def evaluate(params):
        logits = forward(params, eval_x)
        return (
            per_output_bce(logits, eval_y),
            jnp.mean((logits > 0) == (eval_y > 0.5), axis=0),
        )

    def run_eval(step):
        if st["eval_steps"] and st["eval_steps"][-1] >= step:
            return  # already evaluated at this step before checkpointing
        pol, poa = jax.device_get(evaluate(st["params"]))
        st["eval_steps"].append(step)
        st["per_out_loss"].append(pol)
        st["per_out_acc"].append(poa)

    start, t0, executed = st["step"], time.perf_counter(), 0
    pbar = tqdm(range(start, cfg.steps), initial=start, total=cfg.steps,
                desc=cfg.name, disable=quiet)
    for step in pbar:
        if step % cfg.eval_every == 0:
            run_eval(step)
        if step % cfg.checkpoint_every == 0 and step > start:
            st["step"], st["wall_time"] = step, st["wall_time"] + time.perf_counter() - t0
            t0 = time.perf_counter()
            _save_ckpt(cfg.ckpt_path, st)
        if stop_after is not None and executed >= stop_after:
            st["step"], st["wall_time"] = step, st["wall_time"] + time.perf_counter() - t0
            _save_ckpt(cfg.ckpt_path, st)
            pbar.close()
            return None
        st["params"], st["opt_state"], loss = train_step(
            st["params"], st["opt_state"], step
        )
        st["train_loss"][step] = float(loss)
        executed += 1
        if step % 100 == 0:
            pbar.set_postfix(train=f"{st['train_loss'][step]:.4f}")
    run_eval(cfg.steps)
    st["wall_time"] += time.perf_counter() - t0

    np.savez_compressed(
        cfg.npz_path,
        config=json.dumps(asdict(cfg)),
        train_loss=st["train_loss"],
        eval_steps=np.array(st["eval_steps"]),
        per_out_loss=np.array(st["per_out_loss"], dtype=np.float32),
        per_out_acc=np.array(st["per_out_acc"], dtype=np.float32),
        out_depths=np.asarray(circuit.out_depths),
        wall_time=st["wall_time"],
    )
    cfg.ckpt_path.unlink(missing_ok=True)
    if not quiet:
        print(f"[done] {cfg.name}: eval loss {st['per_out_loss'][-1].mean():.4f}, "
              f"acc {st['per_out_acc'][-1].mean():.4f}, {st['wall_time']/60:.1f} min")
    return cfg.npz_path


def load_run(path):
    """-> (config dict, dict of arrays)."""
    d = np.load(path)
    return json.loads(str(d["config"])), {k: d[k] for k in d.files if k != "config"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for f, t in [
        ("width", int), ("mlp_depth", int), ("hidden_ratio", int), ("batch", int),
        ("steps", int), ("lr", float), ("warmup", int), ("schedule", str),
        ("n_wires", int), ("circ_depth", int), ("circuit_seed", int),
        ("model_seed", int), ("data_seed", int), ("eval_seed", int),
        ("eval_every", int), ("eval_n", int), ("checkpoint_every", int),
        ("out_dir", str),
        ("output_wires", lambda s: tuple(int(x) for x in s.split(","))),
    ]:
        default = getattr(RunConfig, f)
        p.add_argument(f"--{f.replace('_', '-')}", type=t, default=default)
    run(RunConfig(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
