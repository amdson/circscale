"""Headless driver for the iterative generalization experiments on the tiny
circuit (see EXPERIMENT_BRIEF.md). Everything lands in runs/iter and is
idempotent (train.run skips done runs and resumes checkpoints).

  uv run python iter_grid.py run <batch> [<batch> ...]   # train + summarize
  uv run python iter_grid.py summary [substring]         # table over runs/iter
  uv run python iter_grid.py agg [substring]             # seed-aggregated table
  uv run python iter_grid.py fig <batch> [<batch> ...]   # figs/iter_<batch>.png
  uv run python iter_grid.py list                        # batches and their runs

Batches are named lists of RunConfig overrides on top of BASE (w128d4, Adam
lr 3e-3, 10k steps, the fixed low-data tiny-circuit task).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from train import RunConfig, load_run, run

TARGET_WIRE = 0
CHANCE = float(np.log(2))
OUT_DIR = "runs/iter"
FIG_DIR = "figs"

BASE = dict(
    width=128, mlp_depth=4, lr=3e-3, steps=10_000,
    n_wires=8, circ_depth=4, circuit_seed=2, output_wires=(TARGET_WIRE,),
    train_frac=0.5, eval_every=100, checkpoint_every=2500, out_dir=OUT_DIR,
)


def cfg(**kw) -> RunConfig:
    return RunConfig(**{**BASE, **kw})


def _grid(**axes):
    """Cartesian product of keyword lists -> list of override dicts."""
    out = [{}]
    for k, vals in axes.items():
        out = [{**o, k: v} for o in out for v in vals]
    return out


SEEDS = [0, 1, 2]

BATCHES: dict[str, list[dict]] = {
    # 1. diagnose (seed 0): clean Adam vs init-relative noise
    "diag": [dict(), dict(weight_noise=0.1), dict(weight_noise=0.3)],
    # 2. Omnigrok lever (seed 0): init scale, plain Adam
    "init": _grid(init_scale=[0.2, 0.3, 0.5, 2.0]),
    # --- seed-replicated batches ---
    "base3": _grid(model_seed=SEEDS),
    "noise3": _grid(weight_noise=[0.15, 0.2, 0.3, 0.5, 0.7, 1.0], model_seed=SEEDS),
    "persist3": _grid(weight_noise=[0.1, 0.3], noise_mode=["persist"], model_seed=SEEDS),
    "absnoise": _grid(weight_noise=[0.01, 0.03], noise_scale=["abs"]),
    "init3": _grid(init_scale=[0.05, 0.1, 0.2], model_seed=SEEDS),
    "adamw3": _grid(optimizer=["adamw"], weight_decay=[1e-2, 1e-1, 3e-1, 1.0],
                    model_seed=SEEDS),
    "eps3": _grid(adam_eps=[1e-4], model_seed=SEEDS),
    # --- scalability of the noise recipe ---
    "lowdata": _grid(train_frac=[0.4, 0.3, 0.25, 0.2], weight_noise=[0.0, 0.5, 1.0],
                     model_seed=SEEDS),
    "noise_x_init": _grid(weight_noise=[0.5], init_scale=[0.2, 0.5], model_seed=SEEDS),
    "sgdnoise": _grid(optimizer=["sgd"], lr=[0.1, 0.3], weight_noise=[0.0, 0.5, 1.0]),
    "circuits": _grid(circuit_seed=[0, 1, 3, 4], weight_noise=[0.0, 0.5]),
    "allwires": _grid(output_wires=[None], weight_noise=[0.0, 0.5], model_seed=SEEDS),
    "shapes_a": [dict(width=w, mlp_depth=d, lr=3e-3, weight_noise=wn)
                 for w, d in [(32, 2), (64, 3), (180, 5), (256, 6), (360, 7), (512, 8)]
                 for wn in [0.0, 0.5]],
    "shapes_b": [dict(width=w, mlp_depth=d, lr=1e-3, weight_noise=wn)
                 for w, d in [(32, 2), (64, 3), (180, 5), (256, 6), (360, 7), (512, 8)]
                 for wn in [0.0, 0.5]],
    # recipe robustness: lr sensitivity, other train pools, long horizon
    "lr": _grid(lr=[1e-3, 1e-2], weight_noise=[0.0, 0.5], model_seed=SEEDS),
    "pools": _grid(pool_seed=[1, 2], weight_noise=[0.0, 0.5], model_seed=SEEDS),
    "long": _grid(steps=[50_000], weight_noise=[0.3, 0.5, 1.0], model_seed=[0, 1]),
    # deep shapes: does the noise recipe need only a lower lr / longer horizon?
    "deep": [dict(width=w, mlp_depth=d, lr=1e-3, steps=30_000, weight_noise=wn)
             for w, d in [(256, 6), (360, 7), (512, 8)] for wn in [0.3, 0.5]],
    # late-collapse fixes (Adam eps, rms-relative noise, wd) on long horizons
    "stab": (_grid(steps=[50_000], weight_noise=[0.5], adam_eps=[1e-6, 1e-4], model_seed=[0, 1])
             + _grid(steps=[50_000], weight_noise=[0.5], noise_scale=["rms"], model_seed=[0, 1])
             + _grid(steps=[50_000], weight_noise=[0.5], optimizer=["adamw"],
                     weight_decay=[0.01], model_seed=[0, 1])),
    "deep_stab": [dict(width=360, mlp_depth=7, lr=1e-3, steps=30_000, weight_noise=0.5, adam_eps=1e-4),
                  dict(width=360, mlp_depth=7, lr=1e-3, steps=30_000, weight_noise=0.5, noise_scale="rms"),
                  dict(width=360, mlp_depth=7, lr=1e-3, steps=30_000, weight_noise=0.5,
                       optimizer="adamw", weight_decay=0.01),
                  dict(width=256, mlp_depth=6, lr=1e-3, steps=30_000, weight_noise=0.3, adam_eps=1e-4),
                  dict(width=256, mlp_depth=6, lr=1e-3, steps=30_000, weight_noise=0.3, noise_scale="rms")],
    # candidate recipe at depth: noise 0.5 + adam_eps 1e-4 + lr 1e-3, seeds
    "deep_eps_a": [dict(width=512, mlp_depth=8, lr=1e-3, steps=30_000, weight_noise=0.5,
                        adam_eps=1e-4, model_seed=s) for s in [0, 1]],
    "deep_eps_b": [dict(width=360, mlp_depth=7, lr=1e-3, steps=30_000, weight_noise=0.5,
                        adam_eps=1e-4, model_seed=s) for s in [1, 2]],
    "deep_eps_c": [dict(width=256, mlp_depth=6, lr=1e-3, steps=30_000, weight_noise=0.5,
                        adam_eps=1e-4, model_seed=s) for s in [0, 1]]
                  + _grid(train_frac=[0.3], weight_noise=[0.5, 1.0], adam_eps=[1e-4], model_seed=SEEDS),
    # scale x noise strength in the low-data regime (params saved for ensembles)
    **{f"scale_w{w}": _grid(width=[w], mlp_depth=[4], lr=[1e-3], steps=[20_000],
                            adam_eps=[1e-4], train_frac=[0.25],
                            weight_noise=[0.5, 1.0, 2.0, 3.0], model_seed=SEEDS,
                            out_dir=["runs/scale"], save_params=[True])
       for w in [64, 128, 256, 512]},
    # round 2: making width help at train_frac 0.25
    "scale2_a": _grid(width=[512], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                      train_frac=[0.25], weight_noise=[0.2, 0.3], model_seed=[0, 1],
                      out_dir=["runs/scale"], save_params=[True])
                + _grid(width=[512], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                        train_frac=[0.25], weight_noise=[0.5, 1.0], noise_scale=["rms"],
                        model_seed=[0, 1], out_dir=["runs/scale"], save_params=[True]),
    "scale2_b": _grid(width=[512], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                      train_frac=[0.25], weight_noise=[0.5], init_scale=[0.5], model_seed=[0, 1],
                      out_dir=["runs/scale"], save_params=[True])
                + _grid(width=[512], mlp_depth=[4], lr=[3e-4], steps=[20_000], adam_eps=[1e-4],
                        train_frac=[0.25], weight_noise=[0.5], model_seed=[0, 1],
                        out_dir=["runs/scale"], save_params=[True]),
    "scale2_c": _grid(width=[64], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                      train_frac=[0.25], weight_noise=[1.0, 2.0], noise_scale=["rms"],
                      model_seed=SEEDS, out_dir=["runs/scale"], save_params=[True])
                + _grid(width=[256], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                        train_frac=[0.25], weight_noise=[0.2, 0.3], model_seed=SEEDS,
                        out_dir=["runs/scale"], save_params=[True]),
    "scale2_d": _grid(width=[128], mlp_depth=[8], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                      train_frac=[0.25], weight_noise=[0.5, 1.0], model_seed=SEEDS,
                      out_dir=["runs/scale"], save_params=[True])
                + _grid(width=[64], mlp_depth=[8], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                        train_frac=[0.25], weight_noise=[1.0], model_seed=SEEDS,
                        out_dir=["runs/scale"], save_params=[True]),
    "scale2_e": _grid(width=[512, 256], mlp_depth=[4], lr=[3e-3], steps=[20_000], adam_eps=[1e-4],
                      train_frac=[0.25], weight_noise=[1.0], model_seed=[0, 1],
                      out_dir=["runs/scale"], save_params=[True])
                + _grid(width=[512], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                        train_frac=[0.25], weight_noise=[0.5], init_scale=[0.2], model_seed=[0, 1],
                        out_dir=["runs/scale"], save_params=[True]),
    # weight decay x noise (ELBO pairing) across width, train_frac 0.25
    **{f"wd_w{w}_{wd:g}": _grid(width=[w], mlp_depth=[4], lr=[1e-3], steps=[20_000],
                                adam_eps=[1e-4], train_frac=[0.25], optimizer=["adamw"],
                                decay_norms=[False], weight_decay=[wd], weight_noise=[0.5, 1.0],
                                model_seed=[0, 1], out_dir=["runs/scale"], save_params=[True])
       for w in [128, 512] for wd in [0.1, 0.3, 1.0]},
    "wd_w64": _grid(width=[64], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                    train_frac=[0.25], optimizer=["adamw"], decay_norms=[False],
                    weight_decay=[0.3, 1.0], weight_noise=[1.0],
                    model_seed=[0, 1], out_dir=["runs/scale"], save_params=[True]),
    "wd2_w512": _grid(width=[512], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                      train_frac=[0.25], optimizer=["adamw"], decay_norms=[False],
                      weight_decay=[0.3, 1.0], weight_noise=[0.5], noise_scale=["rms"],
                      model_seed=[0, 1], out_dir=["runs/scale"], save_params=[True])
                + _grid(width=[512], mlp_depth=[4], lr=[1e-3], steps=[20_000], adam_eps=[1e-4],
                        train_frac=[0.25], optimizer=["adamw"], decay_norms=[False],
                        weight_decay=[1.0], weight_noise=[0.3], model_seed=[0, 1],
                        out_dir=["runs/scale"], save_params=[True]),
    # init-variance-scaled decay (Gaussian prior at init variance) x noise, cheap w128 grid
    **{f"wdi_w128_{wn:g}": _grid(width=[128], mlp_depth=[4], lr=[1e-3], steps=[20_000],
                                 adam_eps=[1e-4], train_frac=[0.25], optimizer=["adamw"],
                                 wd_scale=["init"], weight_decay=[0.01, 0.03, 0.1, 0.3],
                                 weight_noise=[wn], model_seed=[0, 1],
                                 out_dir=["runs/scale"], save_params=[True])
       for wn in [0.5, 1.0]},
    "deep_seeds": [dict(width=512, mlp_depth=8, lr=1e-3, weight_noise=wn, model_seed=s)
                   for wn in [0.0, 0.5] for s in [1, 2]],
}


def batch_cfgs(name: str) -> list[RunConfig]:
    return [cfg(**o) for o in BATCHES[name]]


# ---------------------------------------------------------------- metrics ---

def spike_count(tr_bce: np.ndarray, factor: float = 10.0, floor: float = 0.02) -> int:
    """Distinct spike events in the (clean) train-pool BCE eval series: a
    checkpoint whose loss exceeds `factor` x the running minimum so far and
    `floor`, counted once per contiguous excursion."""
    tl = np.asarray(tr_bce, dtype=np.float64)
    events, inside, run_min = 0, False, np.inf
    for v in tl:
        hi = v > max(factor * run_min, floor)
        if hi and not inside:
            events += 1
        inside = hi
        run_min = min(run_min, v)
    return events


def summarize(path) -> dict:
    c, r = load_run(path)
    steps = r["eval_steps"]
    sel = slice(None) if c["output_wires"] is None else list(c["output_wires"])
    tr_l = r["per_out_loss_tr"][:, sel].mean(1)
    ho_l = r["per_out_loss_ho"][:, sel].mean(1)
    tr_a = r["per_out_acc_tr"][:, sel].mean(1)
    ho_a = r["per_out_acc_ho"][:, sel].mean(1)
    cross = np.flatnonzero(ho_a >= 0.9)
    solved = np.flatnonzero(ho_a >= 0.999)
    pn = r.get("param_norm")
    best = int(np.argmax(ho_a))
    return dict(
        name=Path(path).stem,
        tr_bce=float(tr_l[-1]), ho_bce=float(ho_l[-1]),
        tr_acc=float(tr_a[-1]), ho_acc=float(ho_a[-1]),
        ho_best=float(ho_a[best]), ho_best_step=int(steps[best]),
        ho_min_bce=float(ho_l.min()),
        cross90=int(steps[cross[0]]) if len(cross) else -1,
        solve=int(steps[solved[0]]) if len(solved) else -1,
        pn0=float(pn[0]) if pn is not None else np.nan,
        pn1=float(pn[-1]) if pn is not None else np.nan,
        spikes=spike_count(tr_l),
        minutes=float(r["wall_time"]) / 60,
    )


COLS = [("name", "<62s"), ("tr_bce", ">8.4f"), ("ho_bce", ">8.4f"),
        ("tr_acc", ">7.3f"), ("ho_acc", ">7.3f"), ("ho_best", ">8.3f"),
        ("ho_best_step", ">8d"), ("cross90", ">8d"), ("solve", ">7d"), ("pn0", ">7.2f"),
        ("pn1", ">7.2f"), ("spikes", ">7d"), ("minutes", ">5.1f")]


def print_table(rows: list[dict]) -> None:
    print("".join(f"{k:{f[0]}{re.match(r'[<>](\d+)', f).group(1)}s}" for k, f in COLS))
    for row in rows:
        print("".join(f"{row[k]:{f}}" for k, f in COLS))


def stage_run(names: list[str]) -> None:
    for name in names:
        rows = []
        for c in batch_cfgs(name):
            run(c, quiet=True)
            rows.append(summarize(c.npz_path))
            print(f"[{name}] done {c.name}", flush=True)
        print(f"\n== batch {name} ==")
        print_table(rows)


def stage_summary(sub: str = "") -> None:
    paths = sorted(p for p in Path(OUT_DIR).glob("*.npz") if sub in p.name)
    print_table([summarize(p) for p in paths])


def stage_agg(sub: str = "") -> None:
    """Seed-aggregated table: group runs by name with the _msN tag removed."""
    groups: dict[str, list[dict]] = {}
    for p in sorted(Path(OUT_DIR).glob("*.npz")):
        if sub not in p.name:
            continue
        groups.setdefault(re.sub(r"_ms\d+$", "", p.stem), []).append(summarize(p))
    print(f"{'config':<58s}{'n':>3s}{'ho_acc':>8s}{'min':>7s}{'max':>7s}"
          f"{'ho_bce':>8s}{'n_perf':>7s}{'cross90':>8s}{'solve':>7s}{'pn0':>7s}{'pn1':>7s}{'spk':>5s}")
    for k, rows in groups.items():
        acc = np.array([r["ho_acc"] for r in rows])
        bce = np.array([r["ho_bce"] for r in rows])
        cr = [r["cross90"] for r in rows]
        cr = [c for c in cr if c >= 0]
        sv = [r["solve"] for r in rows if r["solve"] >= 0]
        print(f"{k.replace('w128_d4_b256_lr0.003_s10000', 'base').replace('_c8x4_tf0.5_ow0_cs2', ''):<58s}"
              f"{len(rows):>3d}{acc.mean():>8.3f}{acc.min():>7.3f}{acc.max():>7.3f}"
              f"{bce.mean():>8.3f}{int((acc >= 0.999).sum()):>7d}"
              f"{(int(np.median(cr)) if cr else -1):>8d}"
              f"{(int(np.median(sv)) if sv else -1):>7d}"
              f"{np.mean([r['pn0'] for r in rows]):>7.2f}"
              f"{np.mean([r['pn1'] for r in rows]):>7.2f}"
              f"{int(np.mean([r['spikes'] for r in rows])):>5d}")


def stage_fig(names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in names:
        cs = [c for c in batch_cfgs(name) if c.npz_path.exists()]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for i, c in enumerate(cs):
            _, r = load_run(c.npz_path)
            m = r["eval_steps"] > 0
            s = r["eval_steps"][m]
            col = f"C{i}"
            lab = c.name.replace("w128_d4_b256_lr0.003_s10000", "base") \
                        .replace("_c8x4_tf0.5_ow0_cs2_ms0", "")
            axes[0].plot(s, np.maximum(r["per_out_loss_tr"][m][:, TARGET_WIRE], 1e-6),
                         "--", color=col, lw=0.8)
            axes[0].plot(s, np.maximum(r["per_out_loss_ho"][m][:, TARGET_WIRE], 1e-6),
                         "-", color=col, lw=1.0, label=lab)
            axes[1].plot(s, r["per_out_acc_ho"][m][:, TARGET_WIRE], color=col, lw=1.0)
            if "param_norm" in r:
                axes[2].plot(s, r["param_norm"][m], color=col, lw=1.0)
        axes[0].axhline(CHANCE, color="gray", ls=":", lw=0.6)
        axes[0].set(xscale="log", yscale="log", xlabel="step",
                    ylabel="BCE (dashed=train, solid=held-out)")
        axes[1].set(xscale="log", xlabel="step", ylabel="held-out acc", ylim=(0.4, 1.02))
        axes[2].set(xscale="log", xlabel="step", ylabel="param norm")
        axes[0].legend(fontsize=6)
        fig.suptitle(f"batch {name}")
        fig.tight_layout()
        Path(FIG_DIR).mkdir(exist_ok=True)
        out = f"{FIG_DIR}/iter_{name}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"wrote {out}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    stage, args = sys.argv[1], sys.argv[2:]
    if stage == "run":
        stage_run(args)
    elif stage == "summary":
        stage_summary(args[0] if args else "")
    elif stage == "agg":
        stage_agg(args[0] if args else "")
    elif stage == "fig":
        stage_fig(args)
    elif stage == "list":
        for k in BATCHES:
            print(k)
            for c in batch_cfgs(k):
                print("   ", c.name, "(done)" if c.npz_path.exists() else "")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
