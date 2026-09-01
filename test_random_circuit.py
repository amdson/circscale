import numpy as np
import pytest

from random_circuit import (
    evaluate_np,
    make_jax_evaluator,
    sample_circuit,
    sample_depths,
    to_dimacs,
)


def all_inputs(n):
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(np.uint8)


@pytest.mark.parametrize("dist", ["full", "uniform"])
def test_bijective_small(dist):
    rng = np.random.default_rng(0)
    n, depth = 9, 4
    c = sample_circuit(rng, n, depth, out_depths=sample_depths(rng, n, depth, dist))
    y = evaluate_np(c, all_inputs(n))
    codes = y @ (1 << np.arange(n))
    assert np.array_equal(np.sort(codes), np.arange(2**n))


def test_jax_matches_numpy():
    rng = np.random.default_rng(1)
    c = sample_circuit(rng, n_wires=256, depth=16)
    x = rng.integers(0, 2, size=(64, 256), dtype=np.uint8)
    y_np = evaluate_np(c, x)
    y_jax = np.asarray(make_jax_evaluator(c)(x))
    assert np.array_equal(y_np, y_jax)


def test_light_cone():
    # A depth-d output can depend on at most 3^d inputs.
    rng = np.random.default_rng(2)
    n, depth = 60, 3
    c = sample_circuit(rng, n, depth)
    support = np.zeros(n, dtype=np.int64)
    for _ in range(20):
        base = rng.integers(0, 2, size=(1, n), dtype=np.uint8)
        flips = np.repeat(base, n, axis=0)
        flips[np.arange(n), np.arange(n)] ^= 1
        changed = evaluate_np(c, flips) != evaluate_np(c, base)
        support = np.maximum(support, changed.sum(axis=0))
    assert np.all(support <= 3**c.out_depths)


def test_frozen_wires():
    # An output tapped at depth d matches the same circuit truncated to d layers.
    rng = np.random.default_rng(3)
    c = sample_circuit(rng, n_wires=48, depth=8)
    x = rng.integers(0, 2, size=(16, 48), dtype=np.uint8)
    full = evaluate_np(c, x)
    state = x.copy()
    for t in range(1, c.depth + 1):
        trunc = type(c)(
            c.n_wires, t, np.minimum(c.out_depths, t),
            c.perms[:t], c.inv_perms[:t], c.tables[:t],
        )
        state = evaluate_np(trunc, x)
        done = c.out_depths <= t
        assert np.array_equal(state[:, done], full[:, done])


def unit_propagate(dimacs, assignment):
    """Propagate from a partial assignment {var: bool}. Returns
    (assignment, conflict)."""
    clauses = [
        [int(t) for t in line.split()[:-1]]
        for line in dimacs.splitlines()[1:]
    ]
    assignment = dict(assignment)
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            unassigned, satisfied = [], False
            for lit in clause:
                val = assignment.get(abs(lit))
                if val is None:
                    unassigned.append(lit)
                elif (lit > 0) == val:
                    satisfied = True
                    break
            if satisfied:
                continue
            if not unassigned:
                return assignment, True
            if len(unassigned) == 1:
                lit = unassigned[0]
                assignment[abs(lit)] = lit > 0
                changed = True
    return assignment, False


def test_dimacs_encoding():
    rng = np.random.default_rng(4)
    n, depth = 9, 3
    c = sample_circuit(rng, n, depth, out_depths=sample_depths(rng, n, depth, "full"))
    x = rng.integers(0, 2, size=(1, n), dtype=np.uint8)
    y = evaluate_np(c, x)[0]
    cnf = to_dimacs(c, y)
    n_vars = int(cnf.splitlines()[0].split()[2])

    # The true preimage propagates to a total, conflict-free assignment.
    asg, conflict = unit_propagate(cnf, {i + 1: bool(x[0, i]) for i in range(n)})
    assert not conflict and len(asg) == n_vars

    # Any single-bit-flipped input yields a conflict.
    for i in range(n):
        wrong = {j + 1: bool(x[0, j]) ^ (j == i) for j in range(n)}
        _, conflict = unit_propagate(cnf, wrong)
        assert conflict
