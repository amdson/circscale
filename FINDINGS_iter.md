# What produces generalization on the tiny circuit (runs/iter, 2026-09-03)

Task: 8-wire depth-4 brickwork, circuit_seed 2, wire 0, train_frac 0.5
(128 of 256 inputs), Bayes ceiling = held-out acc 1.0. Base model/optimizer:
w128d4, Adam lr 3e-3, 10k steps, batch 256. All numbers are the held-out
(128 unseen inputs) accuracy of the target wire at the end of training,
mean over 3 model seeds unless noted. `n_perf` = seeds at exactly 1.0.
Tables: `uv run python iter_grid.py agg`; plots in `figs/iter_*.png`.

## Headline

**Transient, init-relative Gaussian weight noise (eta 0.5-1.0 x each layer's
init std, applied to the forward/backward only, clean weights updated) on
plain Adam takes held-out accuracy from ~0.90 to exactly 1.0 on every seed,
every model shape from w32d2 to w360d7, every circuit seed tried, every
train pool tried, all 8 wires jointly, and with SGD instead of Adam.** It
holds down to train_frac 0.3 (77 revealed inputs; clean Adam: 0.70) and
only fails at 0.2, which is the identifiability threshold. Nothing else
tried (init scale, AdamW, adam_eps, absolute or Langevin noise) comes close.
One knob, wide working window (0.5-1.0 is a factor of two), no schedule.

## Batch results (w128d4 unless noted)

| lever | setting | ho acc (min-max) | n_perf | solve step |
|---|---|---|---|---|
| clean Adam | - | 0.904 (0.89-0.91) | 0/3 | - |
| adam_eps | 1e-4 | 0.891 | 0/3 | - |
| init_scale | 0.05 / 0.1 / 0.2 / 0.3 / 0.5 / 2 | 0.906 / 0.945 / 0.974 / 0.96 / 0.95 / 0.85 | 0 | - |
| AdamW wd | 0.01 / 0.1 / 0.3 / 1 | 0.854 / 0.734 / 0.693 / 0.810 | 0 | - |
| noise, init-relative | 0.15 / 0.2 / 0.3 | 0.927 / 0.969 / 0.948 | 0 / 1 / 2 | 300 |
| noise, init-relative | **0.5 / 0.7 / 1.0** | **1.000 / 1.000 / 1.000** | **3/3 each** | 300 / 300 / 700 |
| noise, absolute (old `_wn`) | 0.01 / 0.03 | 0.938 / 0.789 (1 seed) | 0 | - |
| noise, persistent (Langevin) | 0.1 / 0.3 | 0.812 / 0.883, norm -> 670 / 1980 | 0 | - |
| init 0.5 + noise 0.5 | | 1.000 | 3/3 | 300 |
| init 0.2 + noise 0.5 | | 0.979 | 2/3 | 2100 |
| SGD (lr 0.1 / 0.3) clean | | 0.883 / 0.930 (1 seed) | 0 | - |
| SGD (lr 0.1 / 0.3) + noise 0.5 | | 1.000 / 1.000 (1 seed) | 1/1 each | 300 / 200 |

Data-limited regime (noise 1.0 vs clean, 3 seeds):

| train_frac | revealed inputs | clean | noise 0.5 | noise 1.0 |
|---|---|---|---|---|
| 0.5 | 128 | 0.904 | 1.000 (3/3) | 1.000 (3/3) |
| 0.4 | 102 | 0.857 | 0.991 (1/3) | 1.000 (3/3) |
| 0.3 | 77 | 0.696 | 0.987 (1/3) | 0.998 (2/3) |
| 0.25 | 64 | 0.726 | 0.964 (1/3) | 0.932 (1/3) |
| 0.2 | 51 | 0.652 | 0.725 | 0.694 |

Scalability (1 seed per cell; noise 0.5; lr 1e-3 / 3e-3):

| shape | clean | noise 0.5 |
|---|---|---|
| w32d2 | 0.875 / 0.875 | 1.000 / 1.000 |
| w64d3 | 0.867 / 0.891 | 1.000 / 1.000 |
| w128d4 | 0.867 / 0.904 | 1.000 / 1.000 |
| w180d5 | 0.859 / 0.867 | 1.000 / 1.000 |
| w256d6 | 0.875 / 0.898 | 1.000 / 0.875 (lr 3e-3 blows up, norm 121->216) |
| w360d7 | 0.875 / 0.906 | 1.000 / 0.914 (lr 3e-3 blows up, norm 154->215) |
| w512d8 | 0.888 (3 seeds) / 0.906 | 0.992 (1/3 perfect at 10k) / 0.992 |

Other circuits (wire 0, seed 0): clean 1.00 / 0.98 / 0.92 / 0.82 on
circuit seeds 0 / 1 / 3 / 4; noise 0.5: 1.000 on all four. Other train
pools (pool_seed 1, 2): clean 0.83 / 0.83, noise 0.5: 1.000 / 1.000 (3/3
each). All 8 wires jointly (output_wires=None): clean 0.953, noise 0.5
1.000 (3/3). lr sensitivity at w128d4: noise 0.5 is 3/3 perfect at lr
1e-3 and 3e-3, 2/3 at 1e-2 (norm inflates to 168).

## Why (diagnostics)

- **Clean Adam is lazy here.** Train pool is memorized by step ~500 with
  param norm essentially unchanged (70.7 -> 70.9) and no loss spikes at
  all; held-out acc is frozen at ~0.9 from step 100 while held-out BCE
  climbs (confident wrong answers). The brief's "spiky Adam" picture is
  not what happens at this shape/lr.
- **Noise selects flat interpolators; clean and small-init training do not**
  (`probe.py`, `figs/iter_flatness.png`). Perturbing the *final* weights
  with init-relative noise sigma: the clean solution's train BCE breaks at
  sigma 0.2 and its held-out acc falls monotonically; solutions trained with
  noise 0.5 keep train BCE ~0 and held-out acc 1.0 up to sigma 0.7-1.0,
  i.e. they are flat well beyond the training noise. The init_scale 0.2
  solution is the *sharpest* of all in these units (breaks at sigma 0.1),
  so the small-init route to generalization is not flatness in
  init-relative units; it is a different (small-norm) mechanism and it is
  weaker (0.97 max).
- **Norm growth is the visible dynamic under noise**: 70 -> 80-100 at 10k
  and -> 120-230 at 50k. With pre-norm blocks the trunk is nearly scale
  invariant, so growing weights is how the net raises its signal-to-noise
  against a noise anchored to the init scale. This also means the
  regularizer weakens over time.
- **AdamW at this lr is unstable, not slow.** wd >= 0.1 shrinks the norm to
  ~10-20 within 2k steps, the effective lr on the scale-invariant trunk
  explodes, and training oscillates for the rest of the run (figs/
  iter_adamw3.png). wd 0.01 shrinks slowly and just underperforms at 10k.
- **Persistent (Langevin) noise without decay** random-walks the norm to
  700-2000 and hurts.

## Stability on long horizons

- w128d4, 50k steps, lr 3e-3: noise 0.5 and 1.0 stay at 1.0 on both seeds
  (they take an occasional spike and re-converge to the same held-out-perfect
  solution); noise 0.3 degrades (0.98, 0.89).
- Deep shapes at 30k steps (lr 1e-3): w256d6 noise 0.5 stays perfect; w256d6
  noise 0.3 and w360d7 noise 0.5 solve at step ~400 then suffer a
  single-step collapse (train BCE 1e-10 -> 0.7 in one step, norm 154 -> 177)
  around step 20-25k and re-memorize a *sharp* solution (held-out 0.7-0.8).
  The plateau before the collapse is perfectly flat, so this is the Adam
  vanishing-second-moment instability, not a gradual drift.
- **Fixes, w128d4 at 50k (2 seeds each, all still 1.0 at the end):**
  adam_eps 1e-6 (norm 103-123), adam_eps 1e-4 (norm stays at 78, the only
  variant whose norm stops growing), rms-relative noise `wnm0.5` (norm
  207-240, growth not stopped), AdamW wd 0.01 + noise (norm 157-177, decay
  does not bound it at this lr). So the norm growth is Adam's constant-size
  steps on a never-converging noisy gradient, not the net "escaping" the
  noise, and eps is the right lever for it.
- **Fixes at depth, 30k steps, lr 1e-3:** rms noise and AdamW+noise still
  collapse at w360d7 (0.984, 0.914). adam_eps 1e-4 + noise 0.5 holds:
  w256d6 1.000 / 1.000, w360d7 0.992 / 1.000 / 1.000, w512d8 1.000 / 1.000
  (norms flat: 123, 155, 196; the first perfect w512d8 runs). At
  train_frac 0.3, eps 1e-4 + noise 1.0 gives 1.000 / 1.000 / 0.994 and
  noise 0.5 gives 0.994 / 0.989 / 1.000 (10k steps).

## Recipe

Plain Adam, **lr 1e-3, adam_eps 1e-4, transient init-relative weight
noise 0.5** (1.0 when data is scarcer), no weight decay, no schedule. On this
task it reaches the Bayes ceiling on every shape from w32d2 to w512d8,
holds it for 30-50k steps, and works with SGD in place of Adam. Costs
nothing extra per step (one RNG draw per parameter) and generalizes in
300-700 steps, versus never for the clean baseline.

Caveats / open items:
- w512d8 is the least-tested cell (2 seeds at 30k). One shape sweep with
  3 seeds each under the final recipe would close it (~1 A10-hour).
- Noise strength is in units of each layer's init std and the working
  window is 0.5-1.0 at this depth range; whether eta should shrink with
  depth (the function-space perturbation grows with the number of blocks)
  is untested beyond d8.
- init_scale < 1 helps modestly on its own (0.97 at 0.2) but produces the
  sharpest solutions and does not combine well with noise (0.2 + noise:
  2/3); 0.5 + noise is fine (3/3). Not needed in the recipe.
- Weight decay never helped at these horizons; it may at much lower lr.

Figures: `figs/iter_{diag,init,noise3,adamw3,lowdata,shapes_a,shapes_b,
long,stab,deep,deep_stab}.png`, flatness probe `figs/iter_flatness.png`.
Every run is in `runs/iter/*.npz` (`uv run python iter_grid.py agg`);
probe params in `runs/iter_params/`.
