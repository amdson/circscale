"""Build a 256-wire circuit, benchmark JAX throughput, summarize outputs."""

import time

import numpy as np

from random_circuit import evaluate_np, make_jax_evaluator, sample_circuit, to_dimacs

N, DEPTH, BATCH = 256, 32, 1 << 14

rng = np.random.default_rng(0)
circuit = sample_circuit(rng, n_wires=N, depth=DEPTH)
evaluate = make_jax_evaluator(circuit)

x = rng.integers(0, 2, size=(BATCH, N), dtype=np.uint8)
y = np.asarray(evaluate(x))  # first call compiles
assert np.array_equal(y, evaluate_np(circuit, x))

reps = 20
t0 = time.perf_counter()
for _ in range(reps):
    y = evaluate(x)
y.block_until_ready()
dt = time.perf_counter() - t0
print(f"n={N} depth={DEPTH} batch={BATCH}: {reps * BATCH / dt:,.0f} evals/sec (JAX, jitted)")

print(f"output balance: mean={np.asarray(y).mean():.4f} (exact 0.5 over full domain)")

# Sensitivity of outputs grouped by tap depth: fraction of outputs in each
# depth bucket that change under a random single-input-bit flip.
flips = x.copy()
flips[np.arange(BATCH), rng.integers(0, N, BATCH)] ^= 1
changed = np.asarray(evaluate(flips)) != np.asarray(evaluate(x))
print("tap depth -> P(output flips | random single input-bit flip):")
for lo in range(1, DEPTH + 1, 4):
    sel = (circuit.out_depths >= lo) & (circuit.out_depths < lo + 4)
    print(f"  depth {lo:2d}-{lo + 3:2d} ({sel.sum():3d} outputs): {changed[:, sel].mean():.4f}")

cnf = to_dimacs(circuit, np.asarray(evaluate(x[:1]))[0], reveal=np.arange(0, N, 2))
header = cnf.splitlines()[0]
print(f"DIMACS (half of outputs revealed): {header}")
