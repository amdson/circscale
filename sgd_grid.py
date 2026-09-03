"""Headless driver for the SGD regularizer grid on the tiny-circuit task.

Trains the 7-shape model grid with SGD+momentum over all combinations of
weight_decay x weight_noise on the low-data 8-wire circuit (train_frac=0.5:
128 of the 256 inputs), after a short per-shape LR tune. All stages are
idempotent: completed runs are skipped, interrupted runs resume.

  uv run python sgd_grid.py tune     # LR grid per shape (clean runs)
  uv run python sgd_grid.py grid     # shapes x wd x wn at tuned LRs
  uv run python sgd_grid.py figs     # trajectory grid + summary -> figs/
  uv run python sgd_grid.py all      # tune -> grid -> figs
  uv run python sgd_grid.py status   # what's done / checkpointed / pending

Globals are read at call time, so a notebook can override them before
running stages (`import sgd_grid as G; G.SHAPES = [...]`).
`sgd_grid.ipynb` is a thin interactive wrapper around this module.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

from train import RunConfig, load_run, run

# --- task: one tiny circuit, one output wire (matches tiny_circuit.ipynb) ---
N_WIRES, CIRC_DEPTH, CIRCUIT_SEED, TARGET_WIRE = 8, 4, 2, 0
TRAIN_FRAC = 0.5

# --- sweep ---
SHAPES = [(32, 2), (64, 3), (128, 4), (180, 5), (256, 6), (360, 7), (512, 8)]
WD_GRID = [0.0, 1e-2, 1e-1]
WN_GRID = [0.0, 0.03, 0.1]  # init-relative: fraction of each layer's init std
MOMENTUM = 0.9
STEPS = 10_000
LR_GRID = [0.03, 0.1, 0.3, 1.0]
TUNE_STEPS = 1_000
OUT_DIR = "runs/sgd_grid8"
FIG_DIR = "figs"
CHANCE = float(np.log(2))


def base_cfg(width: int, depth: int, lr: float, **kw) -> RunConfig:
    return RunConfig(width=width, mlp_depth=depth, lr=lr,
                     optimizer="sgd", momentum=MOMENTUM,
                     n_wires=N_WIRES, circ_depth=CIRC_DEPTH,
                     circuit_seed=CIRCUIT_SEED, output_wires=(TARGET_WIRE,),
                     train_frac=TRAIN_FRAC, eval_every=100,
                     out_dir=OUT_DIR, **kw)


def n_params(w: int, d: int, hr: int = 4) -> int:
    h = hr * w
    return N_WIRES * w + d * (w + w * h + h * w) + w + w * N_WIRES


def combos() -> list[tuple[float, float]]:
    return [(wd, wn) for wd in WD_GRID for wn in WN_GRID]


def stage_tune() -> dict:
    """Per-shape LR pick: lowest train-pool BCE after TUNE_STEPS clean steps
    (wd = wn = 0). NaN (diverged) counts as infinitely bad."""
    tuned = {}
    for w, d in SHAPES:
        losses = {}
        for lr in LR_GRID:
            cfg = base_cfg(w, d, lr, steps=TUNE_STEPS)
            run(cfg)
            loss = load_run(cfg.npz_path)[1]["per_out_loss_tr"][-1, TARGET_WIRE]
            losses[lr] = np.nan_to_num(loss, nan=np.inf)
        tuned[(w, d)] = best = min(losses, key=losses.get)
        edge = "  ** best at grid edge — extend LR_GRID **" \
            if best in (LR_GRID[0], LR_GRID[-1]) else ""
        print(f"w{w}d{d}: best lr={best:g}   " +
              "  ".join(f"{lr:g}: {l:.4f}" for lr, l in losses.items()) + edge)
    return tuned


def grid_cfgs(tuned: dict) -> dict:
    return {
        (w, d, wd, wn): base_cfg(w, d, tuned[(w, d)], steps=STEPS,
                                 weight_decay=wd, weight_noise=wn)
        for w, d in SHAPES for wd, wn in combos()
    }


def stage_grid(tuned: dict | None = None) -> dict:
    cfgs = grid_cfgs(tuned if tuned is not None else stage_tune())
    print(f"{len(cfgs)} runs")
    for cfg in cfgs.values():
        run(cfg)
    return cfgs


def load_results(cfgs: dict) -> dict:
    return {k: load_run(c.npz_path)[1]
            for k, c in cfgs.items() if c.npz_path.exists()}


def fig_grid(res: dict, save: str | None = None):
    """Rows: shapes. Columns: (wd, wn) combos. Train (dashed) vs held-out
    (solid) target-wire BCE, log-log; losses clipped at 1e-6."""
    import matplotlib.pyplot as plt

    cs = combos()
    fig, axes = plt.subplots(len(SHAPES), len(cs),
                             figsize=(2.4 * len(cs), 2.0 * len(SHAPES)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (w, d) in enumerate(SHAPES):
        for j, (wd, wn) in enumerate(cs):
            ax = axes[i, j]
            r = res.get((w, d, wd, wn))
            if r is None:
                ax.axis("off")
                continue
            m = r["eval_steps"] > 0
            steps = r["eval_steps"][m]
            tr = np.maximum(r["per_out_loss_tr"][m][:, TARGET_WIRE], 1e-6)
            ho = np.maximum(r["per_out_loss_ho"][m][:, TARGET_WIRE], 1e-6)
            ax.plot(steps, tr, "C0--", lw=1.0, label="train")
            ax.plot(steps, ho, "C1-", lw=1.0, label="held-out")
            ax.axhline(CHANCE, color="gray", ls=":", lw=0.6)
            ax.set(xscale="log", yscale="log")
            if i == 0:
                ax.set_title(f"wd={wd:g}\nwn={wn:g}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"w{w}d{d}", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    fig.supxlabel("step")
    fig.supylabel(f"wire {TARGET_WIRE} BCE")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"wrote {save}")
    return fig


def fig_summary(res: dict, save: str | None = None):
    """Final held-out BCE / acc vs params (color = wn, style = wd), plus a
    printed shape x combo accuracy table."""
    import matplotlib.pyplot as plt

    cs = combos()
    Ns = [n_params(w, d) for w, d in SHAPES]
    wd_ls = dict(zip(WD_GRID, ["-", "--", ":", "-."]))
    wn_col = dict(zip(WN_GRID, plt.cm.viridis(np.linspace(0, 0.85, len(WN_GRID)))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for wd, wn in cs:
        bce = [res[(w, d, wd, wn)]["per_out_loss_ho"][-1, TARGET_WIRE]
               if (w, d, wd, wn) in res else np.nan for w, d in SHAPES]
        acc = [res[(w, d, wd, wn)]["per_out_acc_ho"][-1, TARGET_WIRE]
               if (w, d, wd, wn) in res else np.nan for w, d in SHAPES]
        kw = dict(ls=wd_ls[wd], color=wn_col[wn], marker="o", ms=3,
                  label=f"wd={wd:g} wn={wn:g}")
        ax1.plot(Ns, bce, **kw)
        ax2.plot(Ns, acc, **kw)
    ax1.axhline(CHANCE, color="gray", ls=":", lw=0.8)
    ax1.set(xscale="log", yscale="log", xlabel="params N",
            ylabel="final held-out BCE")
    ax2.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax2.set(xscale="log", xlabel="params N", ylabel="final held-out acc",
            ylim=(0.4, 1.02))
    ax1.legend(fontsize=6, ncol=3)
    fig.tight_layout()

    hdr = " " * 8 + "".join(f"wd{wd:g}/wn{wn:g}".rjust(14) for wd, wn in cs)
    print("final held-out acc (target wire)\n" + hdr)
    for w, d in SHAPES:
        row = "".join(
            f"{res[(w, d, wd, wn)]['per_out_acc_ho'][-1, TARGET_WIRE]:>14.3f}"
            if (w, d, wd, wn) in res else f"{'—':>14s}"
            for wd, wn in cs)
        print(f"{f'w{w}d{d}':>8s}{row}")
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"wrote {save}")
    return fig


def stage_figs():
    cfgs = grid_cfgs(stage_tune())
    res = load_results(cfgs)
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    fig_grid(res, save=f"{FIG_DIR}/sgd_grid8_trajectories.png")
    fig_summary(res, save=f"{FIG_DIR}/sgd_grid8_summary.png")


def stage_status():
    def mark(cfg):
        if cfg.npz_path.exists():
            return "done"
        if cfg.ckpt_path.exists():
            with open(cfg.ckpt_path, "rb") as f:
                return f"ckpt@{pickle.load(f)['step']}"
        return "----"

    print("tune:")
    for w, d in SHAPES:
        marks = "  ".join(
            f"lr={lr:g} [{mark(base_cfg(w, d, lr, steps=TUNE_STEPS))}]"
            for lr in LR_GRID)
        print(f"  w{w}d{d:<4} {marks}")
    print("grid (any tuned lr counted):")
    for w, d in SHAPES:
        done = sum(
            base_cfg(w, d, lr, steps=STEPS, weight_decay=wd,
                     weight_noise=wn).npz_path.exists()
            for lr in LR_GRID for wd, wn in combos())
        print(f"  w{w}d{d:<4} {done}/{len(combos())} done")


def main():
    stages = {"tune": stage_tune, "grid": stage_grid, "figs": stage_figs,
              "status": stage_status,
              "all": lambda: (stage_grid(), stage_figs())}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        sys.exit(f"usage: python sgd_grid.py {{{'|'.join(stages)}}}\n\n{__doc__}")
    if sys.argv[1] in ("figs", "all"):
        import matplotlib
        matplotlib.use("Agg")
    stages[sys.argv[1]]()


if __name__ == "__main__":
    main()
