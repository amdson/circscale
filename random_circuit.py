"""Random reversible brickwork circuits with varied-depth output taps.

Model
-----
n wires carry one bit each. The circuit is built in `depth` layers. In each
layer the currently *active* wires are randomly partitioned into triples, and
each triple gets an independent uniformly random 3-bit reversible gate (a
random permutation of {0,1}^3). Each output wire i is assigned a tap depth
d_i in [1, depth]; wire i participates in layers 1..d_i and is frozen
afterwards, so the final state of wire i is its value at depth d_i.

Every layer is a bijection on the full 256-bit state (frozen wires map by
identity), so outputs stay exactly balanced and depth translates into "how
scrambled", not "how degenerate". Precedent: random reversible circuits as
pseudorandom permutations (Gowers 1996; He & O'Donnell 2024) and the layered
architecture of random circuit sampling.

Internally each layer is stored as a wire permutation plus a stack of gate
tables. Slots 0..3G-1 of the permuted state form G triples (G = n // 3);
layers with fewer than G real gates are padded with identity gates acting on
frozen/leftover wires, so every layer has identical shape and evaluation is a
fixed-size gather -> table lookup -> scatter, scanned over layers in JAX.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Circuit:
    n_wires: int
    depth: int
    out_depths: np.ndarray  # (n,) int, tap depth of each output wire, in [1, depth]
    perms: np.ndarray       # (depth, n) int32; slot j of layer t holds wire perms[t, j]
    inv_perms: np.ndarray   # (depth, n) int32; argsort of perms per layer
    tables: np.ndarray      # (depth, G, 8) uint8; identity rows are padding gates

    @property
    def n_gate_slots(self) -> int:
        return self.tables.shape[1]


def sample_depths(
    rng: np.random.Generator, n_wires: int, depth: int, dist: str = "uniform"
) -> np.ndarray:
    """Sample per-output tap depths in [1, depth].

    "uniform": uniform over [1, depth] — a linear ramp of output hardness.
    "full": every output at max depth (the circuit is then a single random
    reversible permutation of {0,1}^n).
    """
    if dist == "uniform":
        return rng.integers(1, depth + 1, size=n_wires)
    if dist == "full":
        return np.full(n_wires, depth, dtype=np.int64)
    raise ValueError(f"unknown depth distribution {dist!r}")


def sample_circuit(
    rng: np.random.Generator,
    n_wires: int = 256,
    depth: int = 32,
    out_depths: np.ndarray | None = None,
) -> Circuit:
    """Sample a random reversible circuit. `out_depths` defaults to uniform.

    Note: at layers where fewer than 3 wires remain active, no gate can be
    placed and those wires pass through; with uniform depths on n=256 this
    only matters in the last couple of layers when depth is large.
    """
    if out_depths is None:
        out_depths = sample_depths(rng, n_wires, depth)
    out_depths = np.asarray(out_depths)
    if out_depths.shape != (n_wires,) or out_depths.min() < 1 or out_depths.max() > depth:
        raise ValueError("out_depths must be (n_wires,) with values in [1, depth]")

    n_slots = n_wires // 3
    perms = np.empty((depth, n_wires), dtype=np.int32)
    tables = np.tile(np.arange(8, dtype=np.uint8), (depth, n_slots, 1))

    for t in range(1, depth + 1):
        active = np.flatnonzero(out_depths >= t)
        rng.shuffle(active)
        g = len(active) // 3
        rest = np.concatenate([active[3 * g :], np.flatnonzero(out_depths < t)])
        rng.shuffle(rest)
        perms[t - 1, : 3 * g] = active[: 3 * g]
        perms[t - 1, 3 * g :] = rest
        if g:
            tables[t - 1, :g] = rng.permuted(
                np.tile(np.arange(8, dtype=np.uint8), (g, 1)), axis=1
            )

    inv_perms = np.argsort(perms, axis=1).astype(np.int32)
    return Circuit(n_wires, depth, out_depths, perms, inv_perms, tables)


def _apply_tables_np(triples: np.ndarray, tables: np.ndarray) -> np.ndarray:
    """triples: (B, G, 3) bits, tables: (G, 8) -> (B, G, 3) output bits."""
    idx = triples[:, :, 0] * 4 + triples[:, :, 1] * 2 + triples[:, :, 2]
    out = tables[np.arange(tables.shape[0]), idx.astype(np.int64)]
    return (out[:, :, None] >> np.array([2, 1, 0], dtype=np.uint8)) & 1


def evaluate_np(circuit: Circuit, inputs: np.ndarray) -> np.ndarray:
    """Reference evaluator. inputs: (B, n) in {0,1} -> outputs (B, n) uint8."""
    state = np.ascontiguousarray(inputs, dtype=np.uint8)
    if state.ndim != 2 or state.shape[1] != circuit.n_wires:
        raise ValueError(f"inputs must be (B, {circuit.n_wires})")
    ng = 3 * circuit.n_gate_slots
    for perm, inv, tab in zip(circuit.perms, circuit.inv_perms, circuit.tables):
        s = state[:, perm]
        bits = _apply_tables_np(s[:, :ng].reshape(-1, circuit.n_gate_slots, 3), tab)
        s[:, :ng] = bits.reshape(-1, ng)
        state = s[:, inv]
    return state


def make_jax_evaluator(circuit: Circuit):
    """Return a jitted function (B, n) uint8 -> (B, n) uint8.

    Recompiles once per distinct batch shape; reuse one batch size when
    sampling many times.
    """
    import jax
    import jax.numpy as jnp

    perms = jnp.asarray(circuit.perms)
    inv_perms = jnp.asarray(circuit.inv_perms)
    tables = jnp.asarray(circuit.tables)
    n_slots = circuit.n_gate_slots
    ng = 3 * n_slots
    gate_arange = jnp.arange(n_slots)
    bit_shifts = jnp.array([2, 1, 0], dtype=jnp.uint8)

    def layer_step(state, layer):
        perm, inv, tab = layer
        s = state[:, perm]
        triples = s[:, :ng].reshape(-1, n_slots, 3)
        idx = (triples[:, :, 0] * 4 + triples[:, :, 1] * 2 + triples[:, :, 2]).astype(
            jnp.int32
        )
        out = tab[gate_arange, idx]
        bits = (out[:, :, None] >> bit_shifts) & 1
        s = s.at[:, :ng].set(bits.reshape(-1, ng))
        return s[:, inv], None

    @jax.jit
    def evaluate(inputs):
        state = inputs.astype(jnp.uint8)
        state, _ = jax.lax.scan(layer_step, state, (perms, inv_perms, tables))
        return state

    return evaluate


def _is_identity_gate(table: np.ndarray) -> bool:
    return bool(np.array_equal(table, np.arange(8)))


def to_dimacs(
    circuit: Circuit,
    outputs: np.ndarray,
    reveal: np.ndarray | None = None,
) -> str:
    """Encode "circuit(x) = outputs on the revealed wires" as DIMACS CNF.

    Variables 1..n are the input bits x_1..x_n; each gate introduces fresh
    variables for its three output wires. Each gate contributes 24 clauses
    (for each of its 8 input patterns, 3 clauses forcing the output bits).
    `reveal` is an index array of output wires to fix (default: all).

    Caveat: with all outputs revealed the instance is polynomial-time
    invertible (run the permutation backwards), though CDCL solvers do not
    automatically exploit this. Reveal a subset of outputs to break
    backward-determinism.
    """
    outputs = np.asarray(outputs).reshape(circuit.n_wires)
    if reveal is None:
        reveal = np.arange(circuit.n_wires)

    wire_var = list(range(1, circuit.n_wires + 1))
    next_var = circuit.n_wires + 1
    clauses: list[str] = []

    for perm, tabs in zip(circuit.perms, circuit.tables):
        for k in range(circuit.n_gate_slots):
            if _is_identity_gate(tabs[k]):
                continue
            wires = perm[3 * k : 3 * k + 3]
            in_vars = [wire_var[w] for w in wires]
            out_vars = list(range(next_var, next_var + 3))
            next_var += 3
            for w, v in zip(wires, out_vars):
                wire_var[w] = v
            for a in range(8):
                antecedent = [
                    -v if (a >> (2 - j)) & 1 else v for j, v in enumerate(in_vars)
                ]
                y = int(tabs[k][a])
                for j, v in enumerate(out_vars):
                    lit = v if (y >> (2 - j)) & 1 else -v
                    clauses.append(" ".join(map(str, antecedent + [lit, 0])))

    for i in reveal:
        v = wire_var[int(i)]
        clauses.append(f"{v if outputs[int(i)] else -v} 0")

    header = f"p cnf {next_var - 1} {len(clauses)}"
    return "\n".join([header, *clauses]) + "\n"
