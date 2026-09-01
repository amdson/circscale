"""Random fan-in-3 tree circuits with interior taps.

A tree over `n_leaves = 3^depth` input bits: level L (1..depth) has
`n_leaves / 3^L` gates, each a uniformly random *balanced* 3-bit -> 1-bit
function (truth table with exactly four 1s) applied to three consecutive
wires of the previous level. Subtrees have disjoint supports, so gate inputs
are independent uniform bits and every interior wire is exactly balanced —
no bias drift, no reversibility needed.

The interior wires (levels 1..depth, `(n_leaves - 1) / 2` of them) are the
candidate outputs. Tapping them sparsely (train.RunConfig.output_wires)
turns supervision density into the experimental dial: a tapped node's
effective difficulty is exponential in its *gap* to the nearest fully
supervised (or leaf) coverage beneath it — see `supervision_gaps`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TreeCircuit:
    n_leaves: int
    depth: int
    tables: tuple = field(repr=False)  # per level: (n_gates, 8) uint8 bits

    @property
    def level_sizes(self) -> list[int]:
        return [self.n_leaves // 3 ** l for l in range(1, self.depth + 1)]

    @property
    def n_interior(self) -> int:
        return (self.n_leaves - 1) // 2

    @property
    def out_depths(self) -> np.ndarray:
        return np.concatenate(
            [np.full(s, l + 1) for l, s in enumerate(self.level_sizes)]
        )


def sample_tree_circuit(rng: np.random.Generator, n_leaves: int = 729,
                        depth: int = 6) -> TreeCircuit:
    if 3 ** depth != n_leaves:
        raise ValueError(f"n_leaves must be 3^depth, got {n_leaves} != 3^{depth}")
    tables = []
    for l in range(1, depth + 1):
        g = n_leaves // 3 ** l
        tables.append(rng.permuted(
            np.tile(np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8), (g, 1)),
            axis=1,
        ))
    return TreeCircuit(n_leaves, depth, tuple(tables))


def evaluate_np(circuit: TreeCircuit, x: np.ndarray) -> np.ndarray:
    """x: (B, n_leaves) bits -> (B, n_interior) interior wire values,
    ordered level 1 first, root last."""
    vals, outs = np.ascontiguousarray(x, dtype=np.uint8), []
    for tab in circuit.tables:
        t = vals.reshape(vals.shape[0], -1, 3)
        idx = t[:, :, 0] * 4 + t[:, :, 1] * 2 + t[:, :, 2]
        vals = tab[np.arange(tab.shape[0]), idx.astype(np.int64)]
        outs.append(vals)
    return np.concatenate(outs, axis=1)


def make_tree_evaluator(circuit: TreeCircuit):
    """Jitted (B, n_leaves) uint8 -> (B, n_interior) uint8."""
    import jax
    import jax.numpy as jnp

    tables = [jnp.asarray(t) for t in circuit.tables]

    @jax.jit
    def evaluate(x):
        vals, outs = x.astype(jnp.uint8), []
        for tab in tables:
            t = vals.reshape(vals.shape[0], -1, 3)
            idx = (t[:, :, 0] * 4 + t[:, :, 1] * 2 + t[:, :, 2]).astype(jnp.int32)
            vals = tab[jnp.arange(tab.shape[0]), idx]
            outs.append(vals)
        return jnp.concatenate(outs, axis=1)

    return evaluate


def supervision_gaps(circuit: TreeCircuit, tapped: np.ndarray) -> np.ndarray:
    """For every interior wire, the depth of unsupervised computation beneath
    it: gap = 1 + max over children of (0 if child is a leaf or tapped, else
    that child's gap). tapped: bool (n_interior,). A tapped node with gap g
    must be learned as a depth-g function of supervised/leaf features."""
    tapped = np.asarray(tapped, dtype=bool)
    gaps, offset = [], 0
    child_eff = np.zeros(circuit.n_leaves, dtype=np.int64)  # leaves covered
    for size in circuit.level_sizes:
        g = 1 + child_eff.reshape(-1, 3).max(axis=1)
        gaps.append(g)
        child_eff = np.where(tapped[offset:offset + size], 0, g)
        offset += size
    return np.concatenate(gaps)
