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
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from mlp import MLPConfig, forward, init_params, init_stds, per_output_bce
from random_circuit import make_jax_evaluator, sample_circuit
from tree_circuit import make_tree_evaluator, sample_tree_circuit


@dataclass(frozen=True)
class RunConfig:
    # model
    width: int = 256
    mlp_depth: int = 4
    hidden_ratio: int = 4
    # multiply all weight-matrix inits by this (norm scales stay at 1).
    # Small values (<1) are the Omnigrok lever: init norm controls the
    # memorize->generalize delay. Init-relative weight_noise stays anchored
    # to the nominal (unscaled) init stds.
    init_scale: float = 1.0
    # training
    batch: int = 256
    steps: int = 50_000
    lr: float = 1e-3
    warmup: int = 500
    schedule: str = "constant"  # "constant" (honest mid-run L(N,D) points) or "cosine"
    optimizer: str = "adam"     # "adam" | "adamw" | "sgd"
    adam_eps: float = 1e-8      # adam/adamw; raising it (1e-4-ish) tames spikes
    weight_decay: float = 0.0   # adamw (decoupled) and sgd (classic L2-style)
    decay_norms: bool = True    # False: decay weight matrices only (norm gains exempt)
    # "const": one decay rate for every leaf. "init": per-leaf rate
    # weight_decay * (std_w1 / std_leaf)^2, i.e. a Gaussian prior whose
    # variance is each leaf's init variance (w1 gets exactly weight_decay;
    # small-init leaves such as w2 are decayed harder). Implies decay_norms=False.
    wd_scale: str = "const"     # "const" | "init"
    momentum: float = 0.9       # sgd only; 0 = vanilla SGD
    # Gaussian weight-noise regularizer (0 disables).
    # noise_mode — "transient": perturb weights for the forward/backward only,
    #   update the clean weights (minimizes the Gaussian-smoothed loss; with
    #   weight decay this is an ELBO with fixed-variance posterior and
    #   Gaussian prior). "persist": add noise into the weights after each
    #   optimizer step (Langevin-style).
    # noise_scale — "init": weight_noise is a dimensionless ratio; each leaf
    #   gets std weight_noise * (its init std), keeping the function-space
    #   perturbation width-invariant (norm scales get none). "abs":
    #   weight_noise is the raw per-parameter std for every leaf. "rms":
    #   like "init" but relative to each leaf's *current* RMS (stop-grad), so
    #   the regularizer does not weaken as weights grow (scale-free).
    weight_noise: float = 0.0
    noise_mode: str = "transient"  # "transient" | "persist"
    noise_scale: str = "init"      # "init" | "abs" | "rms"
    # circuit / seeds
    task: str = "brickwork"  # "brickwork" | "tree3" (n_wires = 3^circ_depth leaves)
    n_wires: int = 256
    circ_depth: int = 32
    circuit_seed: int = 0
    model_seed: int = 0
    data_seed: int | None = None  # None -> model_seed (vary init AND data order)
    eval_seed: int = 1            # keep fixed across the whole sweep
    # loss restriction: train only on these output wires (None = all 256).
    # Eval still records every output, so incidental learning is visible.
    output_wires: tuple | None = None
    # low-data regime: train on a fixed random subset of the input space
    # (requires n_wires <= 16 so it can be enumerated). Eval then runs on the
    # full enumeration and additionally records train-pool (_tr) and held-out
    # (_ho) per-output series. None = online training on fresh samples.
    train_frac: float | None = None
    pool_seed: int = 0
    # large-n variant of the low-data regime: train on a fixed pool of
    # train_n inputs sampled uniformly (pool_seed). Held-out eval uses
    # eval_n fresh inputs (disjoint from the pool w.h.p. for n_wires >~ 30);
    # per_out_loss/acc report the held-out set, plus _tr/_ho series as with
    # train_frac. Mutually exclusive with train_frac.
    train_n: int | None = None
    # "iid": each batch drawn from the pool with replacement. "epoch":
    # pass through the pool in a per-epoch shuffled order, so every sample
    # is seen exactly once per epoch (steps = epochs * n_pool / batch;
    # requires train_frac, and batch must divide the pool size).
    data_order: str = "iid"
    # eval / io
    eval_every: int = 500
    eval_n: int = 4000
    checkpoint_every: int = 2500
    out_dir: str = "runs"
    save_params: bool = False  # also write <name>.params.pkl at the end (not in name)

    @property
    def name(self) -> str:
        s = f"w{self.width}_d{self.mlp_depth}_b{self.batch}_lr{self.lr:g}_s{self.steps}"
        if self.schedule != "constant":
            s += f"_{self.schedule}"
        if self.optimizer != "adam":
            s += f"_{self.optimizer}wd{self.weight_decay:g}"
            if self.optimizer == "sgd" and self.momentum != 0.9:
                s += f"m{self.momentum:g}"
            if self.wd_scale == "init":
                s += "i"
            elif not self.decay_norms:
                s += "wm"
        if self.adam_eps != 1e-8 and self.optimizer in ("adam", "adamw"):
            s += f"_eps{self.adam_eps:g}"
        if self.init_scale != 1.0:
            s += f"_is{self.init_scale:g}"
        if self.weight_noise:  # "wnr" = init-relative, "wn" = absolute, "wnm" = rms-relative; "p" = persist
            tag = {"init": "wnr", "abs": "wn", "rms": "wnm"}[self.noise_scale]
            s += f"_{tag}{self.weight_noise:g}" + ("p" if self.noise_mode == "persist" else "")
        if self.task != "brickwork":
            s += f"_{self.task}"
        if (self.n_wires, self.circ_depth) != (256, 32):
            s += f"_c{self.n_wires}x{self.circ_depth}"
        if self.train_frac is not None:
            s += f"_tf{self.train_frac:g}"
        if self.train_n is not None:
            s += f"_tn{self.train_n}"
        if self.train_frac is not None or self.train_n is not None:
            if self.pool_seed != 0:
                s += f"ps{self.pool_seed}"
            if self.data_order != "iid":
                s += f"_{self.data_order}"
        if self.output_wires is not None:
            ow = "-".join(map(str, self.output_wires))
            if len(self.output_wires) > 12:
                ow = f"{len(self.output_wires)}x{zlib.crc32(ow.encode()):08x}"
            s += f"_ow{ow}"
        return s + f"_cs{self.circuit_seed}_ms{self.model_seed}"

    @property
    def npz_path(self) -> Path:
        return Path(self.out_dir) / f"{self.name}.npz"

    @property
    def ckpt_path(self) -> Path:
        return Path(self.out_dir) / f"{self.name}.ckpt.pkl"

    @property
    def params_path(self) -> Path:
        return Path(self.out_dir) / f"{self.name}.params.pkl"


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


def _make_opt(cfg: RunConfig, mlp_cfg: MLPConfig | None = None):
    sched = _make_schedule(cfg)
    mask = None
    stds = init_stds(mlp_cfg)
    if not cfg.decay_norms or cfg.wd_scale == "init":  # only leaves with init std > 0
        mask = jax.tree_util.tree_map(lambda s: s > 0, stds)
    if cfg.wd_scale == "init":
        ref = stds["blocks"]["w1"] ** 2
        rates = jax.tree_util.tree_map(
            lambda s: cfg.weight_decay * ref / (s * s) if s > 0 else 0.0, stds)
        decay = optax.stateless(lambda upd, params: jax.tree_util.tree_map(
            lambda u, p, d: u + d * p, upd, params, rates))
    elif cfg.wd_scale == "const":
        decay = optax.add_decayed_weights(cfg.weight_decay, mask=mask)
    else:
        raise ValueError(f"unknown wd_scale {cfg.wd_scale!r}")
    if cfg.optimizer == "adam":
        return optax.adam(sched, eps=cfg.adam_eps)
    if cfg.optimizer == "adamw":  # = optax.adamw with a per-leaf decay stage
        return optax.chain(
            optax.scale_by_adam(eps=cfg.adam_eps), decay,
            optax.scale_by_learning_rate(sched))
    if cfg.optimizer == "sgd":
        return optax.chain(decay, optax.sgd(sched, momentum=cfg.momentum or None))
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _epoch_indices(data_key, step: int, batch: int, n_pool: int):
    """Pool indices for one batch under "epoch" ordering: a fresh shuffle of
    the pool per epoch, consumed sequentially — each sample appears exactly
    once per epoch. Works under jit with a traced step."""
    import jax

    epoch = (step * batch) // n_pool
    offset = (step * batch) % n_pool
    perm = jax.random.permutation(jax.random.fold_in(data_key, epoch), n_pool)
    return jax.lax.dynamic_slice(perm, (offset,), (batch,))


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
    if cfg.noise_mode not in ("transient", "persist"):
        raise ValueError(f"unknown noise_mode {cfg.noise_mode!r}")
    if cfg.noise_scale not in ("init", "abs", "rms"):
        raise ValueError(f"unknown noise_scale {cfg.noise_scale!r}")
    if cfg.data_order not in ("iid", "epoch"):
        raise ValueError(f"unknown data_order {cfg.data_order!r}")
    if cfg.train_frac is not None and cfg.train_n is not None:
        raise ValueError("train_frac and train_n are mutually exclusive")
    if cfg.data_order == "epoch" and cfg.train_frac is None and cfg.train_n is None:
        raise ValueError("data_order='epoch' requires train_frac or train_n")
    if cfg.npz_path.exists():
        if not quiet:
            print(f"[skip] {cfg.name} (done)")
        return cfg.npz_path
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    crng = np.random.default_rng(cfg.circuit_seed)
    if cfg.task == "tree3":
        circuit = sample_tree_circuit(crng, cfg.n_wires, cfg.circ_depth)
        circ_eval = make_tree_evaluator(circuit)
        n_out = circuit.n_interior
    elif cfg.task == "brickwork":
        circuit = sample_circuit(crng, cfg.n_wires, cfg.circ_depth)
        circ_eval = make_jax_evaluator(circuit)
        n_out = cfg.n_wires
    else:
        raise ValueError(f"unknown task {cfg.task!r}")
    mlp_cfg = MLPConfig(
        n_inputs=cfg.n_wires, n_outputs=n_out,
        width=cfg.width, depth=cfg.mlp_depth, hidden_ratio=cfg.hidden_ratio,
    )
    opt = _make_opt(cfg, mlp_cfg)
    wire_mask = None if cfg.output_wires is None else jnp.asarray(cfg.output_wires)
    data_key = jax.random.key(cfg.model_seed if cfg.data_seed is None else cfg.data_seed)
    noise_key = jax.random.split(data_key)[0]  # stream independent of the batch keys
    noise_stds = jax.tree_util.tree_leaves(
        init_stds(mlp_cfg) if cfg.noise_scale in ("init", "rms")
        else jax.tree_util.tree_map(lambda _: 1.0, init_stds(mlp_cfg))
    )  # for "rms" these only mark which leaves get noise (init std > 0)

    if cfg.train_frac is not None:
        if cfg.n_wires > 16:
            raise ValueError("train_frac requires n_wires <= 16 (enumerable inputs)")
        n_all = 2 ** cfg.n_wires
        all_x = ((np.arange(n_all)[:, None] >> np.arange(cfg.n_wires)[::-1]) & 1
                 ).astype(np.uint8)
        n_pool = int(round(cfg.train_frac * n_all))
        pool_idx = np.sort(np.random.default_rng(cfg.pool_seed)
                           .choice(n_all, n_pool, replace=False))
        train_x = jnp.asarray(all_x[pool_idx])
    elif cfg.train_n is not None:
        n_pool = cfg.train_n
        train_x = jnp.asarray(np.random.default_rng(cfg.pool_seed).integers(
            0, 2, size=(n_pool, cfg.n_wires), dtype=np.uint8))
    pooled = cfg.train_frac is not None or cfg.train_n is not None
    if pooled and cfg.data_order == "epoch" and n_pool % cfg.batch:
        raise ValueError(f"data_order='epoch' needs batch | n_pool "
                         f"({cfg.batch} does not divide {n_pool})")

    if cfg.ckpt_path.exists():
        with open(cfg.ckpt_path, "rb") as f:
            st = pickle.load(f)
        if not quiet:
            print(f"[resume] {cfg.name} at step {st['step']}")
    else:
        params = init_params(jax.random.key(cfg.model_seed), mlp_cfg)
        if cfg.init_scale != 1.0:  # scale weight matrices only (std > 0)
            params = jax.tree_util.tree_map(
                lambda p, s: p * cfg.init_scale if s else p,
                params, init_stds(mlp_cfg))
        st = {
            "step": 0,
            "param_norm": [],
            "leaf_norms": [],
            "params": params,
            "opt_state": opt.init(params),
            "train_loss": np.zeros(cfg.steps, dtype=np.float32),
            "eval_steps": [],
            "per_out_loss": [],
            "per_out_acc": [],
            "wall_time": 0.0,
        }
        if pooled:
            for k in ("per_out_loss_tr", "per_out_acc_tr",
                      "per_out_loss_ho", "per_out_acc_ho"):
                st[k] = []

    @jax.jit
    def train_step(params, opt_state, step):
        key = jax.random.fold_in(data_key, step)
        if not pooled:
            x = jax.random.bernoulli(
                key, shape=(cfg.batch, cfg.n_wires)
            ).astype(jnp.uint8)
        elif cfg.data_order == "iid":
            x = train_x[jax.random.randint(key, (cfg.batch,), 0, n_pool)]
        else:  # "epoch": every pool sample exactly once per epoch
            x = train_x[_epoch_indices(data_key, step, cfg.batch, n_pool)]
        y = circ_eval(x).astype(jnp.float32)
        def loss_fn(p):
            pol = per_output_bce(forward(p, x), y)
            return jnp.mean(pol if wire_mask is None else pol[wire_mask])

        def add_noise(p_tree):
            leaves, tdef = jax.tree_util.tree_flatten(p_tree)
            keys = jax.random.split(jax.random.fold_in(noise_key, step), len(leaves))
            def std(p, s):
                if cfg.noise_scale != "rms" or s == 0.0:
                    return s
                return jax.lax.stop_gradient(jnp.sqrt(jnp.mean(p * p)))
            return jax.tree_util.tree_unflatten(tdef, [
                p + cfg.weight_noise * std(p, s) * jax.random.normal(k, p.shape, p.dtype)
                for p, s, k in zip(leaves, noise_stds, keys)
            ])

        at = add_noise(params) if cfg.weight_noise and cfg.noise_mode == "transient" \
            else params
        loss, grads = jax.value_and_grad(loss_fn)(at)
        # updates (incl. adamw decay) are applied to the clean weights
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if cfg.weight_noise and cfg.noise_mode == "persist":
            params = add_noise(params)
        return params, opt_state, loss

    if not pooled:
        eval_x = jax.random.bernoulli(
            jax.random.key(cfg.eval_seed), shape=(cfg.eval_n, cfg.n_wires)
        ).astype(jnp.uint8)
    elif cfg.train_frac is not None:
        eval_x = jnp.asarray(all_x)  # full enumeration; _tr/_ho split it
        in_pool = np.zeros(n_all, dtype=bool)
        in_pool[pool_idx] = True
        tr_w = jnp.asarray(in_pool, dtype=jnp.float32)
        ho_w = jnp.asarray(~in_pool, dtype=jnp.float32)
    else:  # train_n: eval on the pool plus eval_n fresh held-out inputs
        ho_x = jax.random.bernoulli(
            jax.random.key(cfg.eval_seed), shape=(cfg.eval_n, cfg.n_wires)
        ).astype(jnp.uint8)
        eval_x = jnp.concatenate([train_x, ho_x])
        w = np.zeros(n_pool + cfg.eval_n, dtype=np.float32)
        w[:n_pool] = 1.0
        tr_w = jnp.asarray(w)
        ho_w = jnp.asarray(1.0 - w)
    eval_y = circ_eval(eval_x).astype(jnp.float32)

    @jax.jit
    def evaluate(params):
        logits = forward(params, eval_x)
        pel = jax.nn.softplus(logits) - logits * eval_y  # per-example BCE
        acc = ((logits > 0) == (eval_y > 0.5)).astype(jnp.float32)
        if cfg.train_n is None:
            out = [jnp.mean(pel, axis=0), jnp.mean(acc, axis=0)]
        else:  # headline series = held-out only (the pool would dilute it)
            out = [(ho_w @ pel) / ho_w.sum(), (ho_w @ acc) / ho_w.sum()]
        if pooled:
            for w in (tr_w, ho_w):
                out += [(w @ pel) / w.sum(), (w @ acc) / w.sum()]
        return out

    eval_keys = ["per_out_loss", "per_out_acc"]
    if pooled:
        eval_keys += ["per_out_loss_tr", "per_out_acc_tr",
                      "per_out_loss_ho", "per_out_acc_ho"]

    def run_eval(step):
        if st["eval_steps"] and st["eval_steps"][-1] >= step:
            return  # already evaluated at this step before checkpointing
        st["eval_steps"].append(step)
        for k, v in zip(eval_keys, jax.device_get(evaluate(st["params"]))):
            st[k].append(v)
        if "param_norm" in st:  # absent when resuming pre-param_norm checkpoints
            sq = [float(jnp.vdot(p, p)) for p in jax.tree_util.tree_leaves(st["params"])]
            st["param_norm"].append(float(np.sqrt(sum(sq))))
            if "leaf_norms" in st:
                st["leaf_norms"].append(np.sqrt(sq))

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

    arrays = {k: np.array(st[k], dtype=np.float32) for k in eval_keys}
    if cfg.train_frac is not None:
        arrays["train_pool"] = pool_idx
    if st.get("param_norm"):
        arrays["param_norm"] = np.array(st["param_norm"], dtype=np.float32)
    if st.get("leaf_norms"):  # (T, n_leaves) in tree_leaves order:
        # embed, norm_scale, w1, w2, final_norm_scale, head
        arrays["leaf_norms"] = np.array(st["leaf_norms"], dtype=np.float32)
    np.savez_compressed(
        cfg.npz_path,
        config=json.dumps(asdict(cfg)),
        train_loss=st["train_loss"],
        eval_steps=np.array(st["eval_steps"]),
        out_depths=np.asarray(circuit.out_depths),
        wall_time=st["wall_time"],
        **arrays,
    )
    if cfg.save_params:
        with open(cfg.params_path, "wb") as f:
            pickle.dump(jax.device_get(st["params"]), f)
    cfg.ckpt_path.unlink(missing_ok=True)
    if not quiet:
        # summarize over the trained wires only, so the number matches the task
        sel = slice(None) if cfg.output_wires is None else list(cfg.output_wires)
        print(f"[done] {cfg.name}: eval loss {st['per_out_loss'][-1][sel].mean():.4f}, "
              f"acc {st['per_out_acc'][-1][sel].mean():.4f}, {st['wall_time']/60:.1f} min")
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
        ("optimizer", str), ("weight_decay", float), ("momentum", float),
        ("task", str),
        ("train_frac", float), ("train_n", int), ("pool_seed", int),
        ("data_order", str),
        ("weight_noise", float), ("noise_mode", str), ("noise_scale", str),
        ("init_scale", float), ("adam_eps", float),
        ("save_params", lambda s: s.lower() in ("1", "true", "yes")),
        ("decay_norms", lambda s: s.lower() in ("1", "true", "yes")),
        ("wd_scale", str),
    ]:
        default = getattr(RunConfig, f)
        p.add_argument(f"--{f.replace('_', '-')}", type=t, default=default)
    run(RunConfig(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
