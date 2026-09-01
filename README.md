# satscaling

Random reversible brickwork circuits with varied-depth output taps, for
sampling (input, output) pairs at scale and generating SAT instances.

## Model

- `n` wires (default 256), `depth` layers. Each layer randomly partitions the
  active wires into triples and applies an independent uniformly random 3-bit
  reversible gate (a random permutation of {0,1}^3) to each triple.
- Each output wire `i` gets a tap depth `d_i` in `[1, depth]` (uniform by
  default): it participates in layers `1..d_i` and is frozen afterwards, so
  output `i` is the wire's value at depth `d_i`. Depth is a per-output
  hardness dial — a depth-`d` output depends on at most `3^d` inputs.
- Every layer is a bijection on the full state, so outputs are exactly
  balanced over the input domain and never degenerate toward constants
  (the failure mode of random *irreversible* circuits, cf. Brodsky &
  Pippenger 2005). Precedent: random reversible circuits as pseudorandom
  permutations (Gowers 1996; He & O'Donnell 2024) and layered random circuit
  sampling.

## Usage

Self-contained uv project (`uv sync` to create `.venv`):

```bash
uv run python demo.py   # throughput + sanity summary
uv run pytest           # tests
```

```python
import numpy as np
from random_circuit import sample_circuit, make_jax_evaluator, to_dimacs

rng = np.random.default_rng(0)
circuit = sample_circuit(rng, n_wires=256, depth=32)   # numpy construction
evaluate = make_jax_evaluator(circuit)                 # jitted, batched
y = evaluate(rng.integers(0, 2, size=(16384, 256), dtype=np.uint8))

cnf = to_dimacs(circuit, np.asarray(y[0]), reveal=np.arange(0, 256, 2))
```

`make_jax_evaluator` compiles once per batch shape (~140k full evaluations/sec
at n=256, depth=32 on CPU); `evaluate_np` is the numpy reference.

## MLP (`mlp.py`)

Pre-norm residual MLP (pure JAX pytree; optax available for training):
`h = h + gelu(rmsnorm(h) @ W1) @ W2` repeated `depth` times over a linear
residual stream of size `width`, bits embedded as +/-1, RMSNorm + linear head
to per-output logits. Init: He (`sqrt(2/fan_in)`) into the GELU,
variance-preserving residual writes scaled by `1/sqrt(depth)` (GPT-2 scheme,
one write per block), `1/sqrt(width)` head for O(1) logits at init.

```python
from mlp import MLPConfig, init_params, forward, bce_loss, per_output_bce
cfg = MLPConfig(width=512, depth=8)   # n_inputs/n_outputs default to 256
params = init_params(jax.random.key(0), cfg)
logits = forward(params, x_bits)      # (B, 256)
```

`per_output_bce` gives a (n_outputs,) loss vector — index it by
`circuit.out_depths` for depth-resolved learning curves.

## Training notebook (`train_circuit.ipynb`)

Reproducibly samples the circuit, then trains the MLP online — every step
draws a fresh batch (infinite data, sampling with replacement), so train loss
is itself a generalization estimate. Defaults: 50k steps, eval on 1000 fixed
held-out samples every 500 steps (~15 min on CPU). Plots train/eval loss,
mean accuracy, and depth-resolved learning curves (accuracy grouped by output
tap depth); saves everything to `history_<run>.npz`. All randomness is
seeded, including the batch stream (batch t = fold_in(DATA_SEED, t)).

## Scaling sweep (`train.py`, `sweep.py`)

`train.py` runs one online-training run (config -> `runs/<name>.npz` with the
full per-output eval time-series; every eval checkpoint is an
(N, D=batch*step) datapoint, honest under the default constant-LR schedule).
Runs checkpoint every 2500 steps and resume exactly (the batch stream is a
pure function of the step index), and completed runs are skipped, so every
stage can be interrupted and re-invoked freely.

```bash
uv run python sweep.py lr-tune   # 5k-step LR grid {1e-4..1e-2} per shape
                                 #   -> runs/tune/ + runs/lr_table.json
uv run python sweep.py main      # 50k-step runs over the scale grid
uv run python sweep.py status    # done / ckpt@step / pending
```

Grid: Kaplan-style co-scaling of width and depth (depth ~ sqrt(width)) at
half-octave width spacing — 9 shapes from (32,2) to (512,8), trunk params
~16k -> ~17M, one seed per shape (noise handled by fitting across the finer
grid rather than replicates; add seeds to `sweep.SEEDS` later if needed).
`sweep.GRID`/`sweep.SEEDS` (and LR_GRID, *_STEPS, BATCH) are module globals
read at call time, so notebooks can override them before invoking the
stages. `train.load_run(path)` -> (config, arrays) for analysis.

## Colab (`colab_sweep.ipynb`)

Runs the entire sweep on a Colab GPU (~30-60 min on a T4): installs CUDA
JAX + optax, clones this repo (edit `REPO_URL`), optionally symlinks `runs/`
into Google Drive so progress survives runtime death, then runs `lr-tune` ->
`main` -> quick-look plots. Every stage resumes, so disconnects only cost up
to `checkpoint_every` steps.

## Notes

- With a random single input-bit flip, shallow-tapped outputs flip with
  probability ~0.01 and the avalanche saturates near 0.5 by tap depth ~16
  (see `demo.py`), so depths beyond ~16 at n=256 give near-fully-mixed
  outputs.
- `to_dimacs` encodes `circuit(x) = y` (24 clauses per gate plus unit clauses
  for revealed outputs). With **all** outputs revealed the instance is
  polynomial-time invertible — the circuit is a permutation, so run it
  backwards; reveal a subset of outputs to break backward-determinism.
- If fewer than 3 wires remain active at some layer, they pass through
  ungated; with uniform depths this only affects the last layer or two.
