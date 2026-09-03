"""Width x noise-strength in the low-data regime (runs/scale): per-cell
held-out accuracy, plus the 3-seed ensemble (majority vote and mean logit)
scored on the same held-out inputs, so a single wide model can be compared
against an ensemble of narrow ones.

  uv run python scale_analysis.py
"""

from __future__ import annotations

import glob
import pickle

import jax.numpy as jnp
import numpy as np

from mlp import forward
from probe import task_arrays
from sgd_grid import n_params
from train import RunConfig, load_run


def main():
    cells: dict[tuple, list] = {}
    for p in sorted(glob.glob("runs/scale/*.npz")):
        c, r = load_run(p)
        cfg = RunConfig(**{k: (tuple(v) if k == "output_wires" and v is not None else v)
                           for k, v in c.items()})
        with open(cfg.params_path, "rb") as f:
            params = pickle.load(f)
        x, y, in_pool = task_arrays(cfg)
        logit = np.asarray(forward(params, x)[:, cfg.output_wires[0]])
        ho = ~np.asarray(in_pool)
        yy = np.asarray(y[:, cfg.output_wires[0]])
        acc_ho = r["per_out_acc_ho"][:, cfg.output_wires[0]]
        solved = np.flatnonzero(acc_ho >= 0.999)
        tag = {"init": "wnr", "abs": "wn", "rms": "wnm"}[c["noise_scale"]]
        extra = (f"_is{c['init_scale']:g}" if c["init_scale"] != 1 else "") + \
                (f"_lr{c['lr']:g}" if c["lr"] != 1e-3 else "") + \
                (f"_wd{c['weight_decay']:g}" + ("i" if c.get("wd_scale") == "init" else "")
                 if c["optimizer"] == "adamw" else "")
        cells.setdefault((c["width"], c["mlp_depth"], f"{tag}{c['weight_noise']:g}{extra}"), []).append(dict(
            seed=c["model_seed"], logit=logit, ho=ho, y=yy,
            acc=float(((logit > 0) == (yy > 0.5))[ho].mean()),
            acc_tail=float(acc_ho[-max(1, len(acc_ho) // 10):].mean()),
            solve=int(r["eval_steps"][solved[0]]) if len(solved) else -1,
            norm_x=float(r["param_norm"][-1] / r["param_norm"][0]),
            tr_bce=float(r["per_out_loss_tr"][-1, cfg.output_wires[0]]),
        ))

    print(f"{'shape':<8}{'params':>9}{'noise':>16}{'n':>3}{'ho_acc':>8}{'min':>7}{'max':>7}"
          f"{'tail':>6}{'n_perf':>7}{'solve':>7}{'norm x':>7}{'tr_bce':>8}{'vote':>7}{'meanlogit':>10}")
    for k in sorted(cells):
        v = cells[k]
        acc = np.array([d["acc"] for d in v])
        sv = [d["solve"] for d in v if d["solve"] >= 0]
        ho, yy = v[0]["ho"], v[0]["y"]
        votes = np.mean([d["logit"] > 0 for d in v], axis=0) > 0.5
        mean_logit = np.mean([d["logit"] for d in v], axis=0) > 0
        vote_acc = float((votes == (yy > 0.5))[ho].mean())
        ml_acc = float((mean_logit == (yy > 0.5))[ho].mean())
        print(f"w{k[0]}d{k[1]:<3}{n_params(k[0], k[1]):>9}{k[2]:>16}{len(v):>3}"
              f"{acc.mean():>8.3f}{acc.min():>7.3f}{acc.max():>7.3f}"
              f"{np.mean([d['acc_tail'] for d in v]):>6.3f}"
              f"{int((acc >= 0.999).sum()):>7}{(int(np.median(sv)) if sv else -1):>7}"
              f"{np.mean([d['norm_x'] for d in v]):>7.2f}"
              f"{np.mean([d['tr_bce'] for d in v]):>8.4f}{vote_acc:>7.3f}{ml_acc:>10.3f}")


if __name__ == "__main__":
    main()
