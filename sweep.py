"""Staged scaling-law sweep driver. All stages are idempotent: completed runs
are skipped, interrupted runs resume from their checkpoint, so ctrl-C at any
point loses at most `checkpoint_every` steps.

  uv run python sweep.py lr-tune   # short-horizon LR grid per (width, depth)
                                   #   -> runs/tune/*.npz + runs/lr_table.json
  uv run python sweep.py main      # full runs: widths at fixed depth
  uv run python sweep.py status    # what's done / checkpointed / pending

Every eval checkpoint of every run is an (N, D=batch*step, loss) datapoint;
the constant-LR schedule keeps mid-run points honest.

Width is the single scale axis at fixed MLP depth, leaning on Kaplan-style
shape-insensitivity (loss depends strongly on N, weakly on aspect ratio). If
per-tap-depth curves later suggest a depth ceiling, add an iso-parameter
shape arm ((512,1)..(128,16) at ~2.1M trunk params) to check.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from train import RunConfig, run

# --- vary width at fixed depth ---
WIDTHS = [32, 64, 128, 256, 512]
MLP_DEPTH = 4
SEEDS = {32: [0, 1, 2], 64: [0, 1, 2], 128: [0, 1, 2], 256: [0, 1], 512: [0]}

LR_GRID = [3e-4, 1e-3, 3e-3]
TUNE_STEPS = 5_000
FULL_STEPS = 50_000
BATCH = 256

TUNE_DIR = "runs/tune"
MAIN_DIR = "runs"
LR_TABLE = Path("runs/lr_table.json")


def all_shapes() -> list[tuple[int, int]]:
    return [(w, MLP_DEPTH) for w in WIDTHS]


def key(width: int, depth: int) -> str:
    return f"w{width}d{depth}"


def tune_cfg(width: int, depth: int, lr: float) -> RunConfig:
    return RunConfig(width=width, mlp_depth=depth, batch=BATCH,
                     steps=TUNE_STEPS, lr=lr, out_dir=TUNE_DIR)


def main_cfg(width: int, depth: int, lr: float, seed: int) -> RunConfig:
    return RunConfig(width=width, mlp_depth=depth, batch=BATCH,
                     steps=FULL_STEPS, lr=lr, model_seed=seed, out_dir=MAIN_DIR)


def final_eval_loss(cfg: RunConfig) -> float:
    return float(np.load(cfg.npz_path)["per_out_loss"][-1].mean())


def read_lr_table() -> dict[str, float]:
    return json.loads(LR_TABLE.read_text()) if LR_TABLE.exists() else {}


def build_lr_table() -> dict[str, float]:
    """Pick the best LR per shape from completed tune runs; warn on grid edges."""
    table = {}
    for width, depth in all_shapes():
        losses = {
            lr: final_eval_loss(tune_cfg(width, depth, lr))
            for lr in LR_GRID
            if tune_cfg(width, depth, lr).npz_path.exists()
        }
        if len(losses) < len(LR_GRID):
            print(f"{key(width, depth)}: {len(losses)}/{len(LR_GRID)} tune runs done, skipping")
            continue
        best = min(losses, key=losses.get)
        table[key(width, depth)] = best
        summary = "  ".join(f"lr={lr:g}: {l:.4f}" for lr, l in losses.items())
        edge = "  ** best at grid edge — consider extending LR_GRID **" \
            if best in (min(LR_GRID), max(LR_GRID)) else ""
        print(f"{key(width, depth)}: best lr={best:g}   ({summary}){edge}")
    if table:
        LR_TABLE.parent.mkdir(parents=True, exist_ok=True)
        LR_TABLE.write_text(json.dumps(table, indent=2))
        print(f"wrote {LR_TABLE}")
    return table


def _tuned_lr(table: dict[str, float], width: int, depth: int) -> float:
    k = key(width, depth)
    if k not in table:
        sys.exit(f"no tuned LR for {k} — finish `sweep.py lr-tune` first")
    return table[k]


def stage_lr_tune():
    for width, depth in all_shapes():
        for lr in LR_GRID:
            run(tune_cfg(width, depth, lr))
    build_lr_table()


def stage_main():
    table = read_lr_table()
    for width in WIDTHS:
        lr = _tuned_lr(table, width, MLP_DEPTH)
        for seed in SEEDS[width]:
            run(main_cfg(width, MLP_DEPTH, lr, seed))


def stage_status():
    def mark(cfg):
        if cfg.npz_path.exists():
            return "done"
        if cfg.ckpt_path.exists():
            with open(cfg.ckpt_path, "rb") as f:
                return f"ckpt@{pickle.load(f)['step']}"
        return "----"

    table = read_lr_table()
    print("lr-tune:")
    for width, depth in all_shapes():
        marks = "  ".join(
            f"lr={lr:g} [{mark(tune_cfg(width, depth, lr))}]" for lr in LR_GRID
        )
        print(f"  {key(width, depth):<9} {marks}")
    print(f"lr_table: {table or 'missing'}")

    def show(label, pairs, seeds_of):
        print(f"{label}:")
        for width, depth in pairs:
            lr = table.get(key(width, depth))
            marks = "  ".join(
                f"ms{s} [{mark(main_cfg(width, depth, lr or 0.0, s))}]"
                for s in seeds_of(width)
            )
            print(f"  {key(width, depth):<9} lr={lr if lr is not None else '?'}  {marks}")

    show("main", all_shapes(), lambda w: SEEDS[w])


if __name__ == "__main__":
    stages = {"lr-tune": stage_lr_tune, "main": stage_main, "status": stage_status}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        sys.exit(f"usage: python sweep.py {{{'|'.join(stages)}}}\n\n{__doc__}")
    stages[sys.argv[1]]()
