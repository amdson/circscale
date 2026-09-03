# Experiment brief: what produces generalization on the tiny circuit?

Mission for a Claude Code session on the GPU VM. Iterate in small batches,
report findings as they come, expand only what works.

## Goal

The user is investigating trading compute for a better Bayesian prior.
Concrete question: on the low-data tiny-circuit task, which training
choices actually produce held-out generalization, and why?

## Task (fixed unless noted)

8-wire brickwork circuit, depth 4, `circuit_seed=2`, solo `output_wires=(0,)`
(note: wire 0's depth-4 tap is effectively depth 2 — layers 3-4 place no
gates at this seed). `train_frac=0.5`: a fixed pool of 128 of the 256
inputs (pool_seed 0); eval enumerates all 256 and records train-pool
(`per_out_loss_tr`) and held-out (`per_out_loss_ho`) series separately.
Identifiability was checked: at 128 revealed inputs the function is
recoverable in principle (no consistent impostor circuits; ambiguity starts
around train_frac ~0.2), so held-out shortfall = network prior +
optimization, not task ambiguity. Bayes ceiling ~= 1.0.

## What's known so far

- Adam (lr 3e-3 at w128d4, from runs/lr_table.json) reaches ~0.8+ held-out
  acc; training shows large loss spikes (slingshot-like).
- Absolute weight noise (old `_wn` runs) helped generalization a lot but
  destabilized training.
- A full SGD+momentum grid over wd x wn (runs/tiny5_sgd + user's own runs)
  was drastically worse than Adam — consistent with "SGD stops moving after
  interpolation; Adam's normalization keeps sculpting features".
- Weight decay at 10k steps often *hurt* short runs; its benefit may be
  horizon-limited.

## Levers in RunConfig (all name-tagged, runs idempotent/resumable)

- `init_scale` (Omnigrok lever, top candidate): multiplies weight-matrix
  inits; norm gains untouched. Small (<1) predicted to generalize fast
  without regularizers.
- `weight_noise` + `noise_mode` ("transient" default = ELBO-style,
  "persist" = Langevin) + `noise_scale` ("init" default: eta as fraction of
  each layer's init std; "abs" = raw std).
- `optimizer` adam | adamw (decoupled wd, `weight_decay`) | sgd (L2 wd,
  `momentum`); `adam_eps` (raise to ~1e-4 to flatten spikes).
- `param_norm` is logged at every eval checkpoint — use it to diagnose
  lazy-vs-rich / norm-shrink stories alongside the tr/ho curves.

## Plan (adapt freely; batch on w128d4 first, minutes per run)

1. Diagnose: Adam clean vs wnr0.1 — do held-out gains coincide with norm
   shrink, growth, or spikes?
2. init_scale in {0.2, 0.3, 0.5, 1, 2}, plain Adam, no regularizers.
3. AdamW wd sweep; extend promising runs to 30-50k steps (delayed benefit).
4. Noise (relative units) x Adam; then interactions of the best two levers.
5. Spike test: best config at adam_eps in {1e-8, 1e-4} — are spikes
   load-bearing (slingshot) or incidental?
6. Winner: +seeds (model_seed), full 7-shape sweep, compare vs Bayes
   ceiling 1.0. If small-init SGD generalizes, that is a headline result.

## Conventions & budget

- Compute budget: ~5 A10-hours total. One 10k-step w128d4 run is ~1 min.
  Check with the user before any single batch that would exceed ~1 GPU-hour.
- Put iterative runs in `out_dir="runs/iter"`. Never delete runs/.
- `uv run python ...` for everything; `uv run pytest -q` after any code
  change; commit+push meaningful code changes (user pulls locally).
- Report per batch: final tr/ho BCE + acc, step where ho acc crosses 0.9,
  param_norm start->end. Plots to figs/ if useful.
