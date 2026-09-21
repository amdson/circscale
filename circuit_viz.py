"""Rough visualizations of the repo's tapped circuits, for sanity checks and figures.

Brickwork (`random_circuit.Circuit`) — both "small" (one target wire, k
relevant inputs) and "rich" (many wires tapped at many depths) tasks:

  draw_circuit    wire diagram. `highlight=` colors the backward light cone of
                  the given output wire(s); `compact=True` drops everything
                  outside it (the "k inputs -> 1 output" picture).
  plot_function   small-task view: Karnaugh map of one output over its
                  relevant inputs + Walsh weight by degree.
  plot_summary    rich-task dashboard: tap-depth histogram, support size vs
                  depth, sensitivity vs depth, output x input influence matrix.
  plot_influence  just the influence-matrix panel.

Tree (`tree_circuit.TreeCircuit`):

  draw_tree       fan-in-3 tree, tapped nodes colored by supervision gap.

Array helpers (no plotting): layer_gates, light_cone, cone_inputs, supports,
influence, restricted_truth_table, walsh_by_degree.
"""

from __future__ import annotations

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from random_circuit import Circuit, evaluate_np
from tree_circuit import TreeCircuit, supervision_gaps

CONE = "#c0392b"      # light-cone gates / highlighted outputs
INK = "#2b2b2b"       # wires and gates when nothing is highlighted
FADED = "#cfcfcf"     # outside the light cone / frozen wire spans


# --------------------------------------------------------------------------
# brickwork structure
# --------------------------------------------------------------------------

def layer_gates(circuit: Circuit) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Real (non-padding) gates per layer as (wires (3,), table (8,)) pairs.

    A slot is a real gate iff all three of its wires are still active at that
    layer; padding slots always contain frozen or leftover wires.
    """
    out = []
    for t in range(circuit.depth):
        active = circuit.out_depths >= t + 1
        layer = []
        for k in range(circuit.n_gate_slots):
            w = circuit.perms[t, 3 * k:3 * k + 3]
            if active[w].all():
                layer.append((w, circuit.tables[t, k]))
        out.append(layer)
    return out


def light_cone(circuit: Circuit, wires, gates=None):
    """Backward light cone of output wire(s).

    Returns (live, gate_live): live[w, t] says the value of wire w after
    layer t (t=0: the input) can influence a chosen output; gate_live[t][j]
    marks gate j of `layer_gates(circuit)[t]` (layer t+1) as inside the cone.
    """
    wires = np.atleast_1d(wires)
    gates = layer_gates(circuit) if gates is None else gates
    live = np.zeros((circuit.n_wires, circuit.depth + 1), dtype=bool)
    gate_live = [np.zeros(len(g), dtype=bool) for g in gates]
    cur = np.zeros(circuit.n_wires, dtype=bool)
    for t in range(circuit.depth, 0, -1):
        cur[wires[circuit.out_depths[wires] == t]] = True
        live[:, t] = cur
        for j, (w, _) in enumerate(gates[t - 1]):  # gates in a layer are disjoint
            if cur[w].any():
                gate_live[t - 1][j] = True
                cur[w] = True
    live[:, 0] = cur
    return live, gate_live


def cone_inputs(circuit: Circuit, wire: int) -> np.ndarray:
    """Input wires structurally connected to output `wire`."""
    return np.flatnonzero(light_cone(circuit, wire)[0][:, 0])


def supports(circuit: Circuit) -> np.ndarray:
    """(n_out, n_in) bool: structural support (light cone) of every output."""
    n = circuit.n_wires
    sup = np.eye(n, dtype=bool)
    for layer in layer_gates(circuit):
        for w, _ in layer:
            sup[w] = sup[w].any(axis=0)
    return sup  # frozen wires keep the support they had at their tap depth


def influence(circuit: Circuit, n_samples: int = 1024, rng=None,
              exact_max_wires: int = 14) -> np.ndarray:
    """(n_out, n_in) influence Inf_i(y_o) = P_x[y_o(x) != y_o(x ^ e_i)].

    Exact (full enumeration) for n_wires <= exact_max_wires, otherwise a
    Monte Carlo estimate from `n_samples` uniform inputs.
    """
    n = circuit.n_wires
    if n <= exact_max_wires:
        x = np.array(list(product([0, 1], repeat=n)), dtype=np.uint8)
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        x = rng.integers(0, 2, size=(n_samples, n), dtype=np.uint8)
    y = evaluate_np(circuit, x)
    inf = np.empty((n, n))
    for i in range(n):
        xf = x.copy()
        xf[:, i] ^= 1
        inf[:, i] = (evaluate_np(circuit, xf) != y).mean(axis=0)
    return inf


def restricted_truth_table(circuit: Circuit, wire: int, max_inputs: int = 20):
    """Enumerate output `wire` over its light-cone inputs (others fixed to 0).

    Returns (inputs, table): inputs are the *functionally* relevant input
    wires (structural cone minus any the gates happen to cancel), table is
    the (2^k,) output indexed with inputs[0] as the most significant bit.
    """
    cone = cone_inputs(circuit, wire)
    if len(cone) > max_inputs:
        raise ValueError(f"wire {wire} has {len(cone)} cone inputs > {max_inputs}")
    assign = np.array(list(product([0, 1], repeat=len(cone))), dtype=np.uint8)
    x = np.zeros((len(assign), circuit.n_wires), dtype=np.uint8)
    x[:, cone] = assign
    y = evaluate_np(circuit, x)[:, wire]
    k = len(cone)
    t = y.reshape((2,) * k) if k else y.reshape(())
    relevant = [a for a in range(k) if np.any(np.take(t, 0, a) != np.take(t, 1, a))]
    t = t[tuple(slice(None) if a in relevant else 0 for a in range(k))]
    return cone[relevant], np.asarray(t).reshape(-1)


def walsh_by_degree(table: np.ndarray) -> np.ndarray:
    """Fourier weight of (-1)^table at each degree 0..k (sums to 1)."""
    k = int(np.log2(len(table)))
    f = 1.0 - 2.0 * table.astype(float)
    h = 1
    while h < len(f):  # in-place fast Walsh-Hadamard transform
        f = f.reshape(-1, 2, h)
        f = np.stack([f[:, 0] + f[:, 1], f[:, 0] - f[:, 1]], axis=1).reshape(-1)
        h *= 2
    w = (f / len(f)) ** 2
    deg = np.array([bin(s).count("1") for s in range(len(f))])
    return np.bincount(deg, weights=w, minlength=k + 1)


# --------------------------------------------------------------------------
# brickwork drawing
# --------------------------------------------------------------------------

def _wire_order(circuit: Circuit, wires: np.ndarray, order) -> np.ndarray:
    if isinstance(order, str):
        if order == "index":
            return wires
        if order == "depth":  # deepest taps on top, stable within a depth
            return wires[np.argsort(-circuit.out_depths[wires], kind="stable")]
        raise ValueError(f"unknown order {order!r}")
    order = np.asarray(order)
    return order[np.isin(order, wires)]


def draw_circuit(circuit: Circuit, highlight=None, compact: bool = False,
                 order="index", show_tables: bool = False, labels=None,
                 ax=None, cmap="viridis", col_width: float | None = None):
    """Wire diagram: time runs left to right, one horizontal line per wire.

    Each 3-bit gate is a vertical bar with dots on its three wires (within a
    layer, gates are staggered so their bars don't overlap). A wire's tap is
    marked where it freezes, colored by tap depth; its frozen remainder is
    drawn faint.

    highlight   output wire(s) whose backward light cone is drawn in color.
    compact     with `highlight`: keep only light-cone wires and gates, and
                stop at the deepest highlighted tap.
    order       "index", "depth", or an explicit wire order (top to bottom).
    show_tables annotate each gate with its truth table (the images of
                inputs 0..7, top wire = most significant bit).
    labels      draw x_i / y_i labels (default: when <= 40 wires are shown).
    """
    gates = layer_gates(circuit)
    n, D = circuit.n_wires, circuit.depth
    if col_width is None:
        col_width = 0.9 if show_tables else 0.45
    if highlight is not None:
        highlight = np.atleast_1d(highlight)
        live, gate_live = light_cone(circuit, highlight, gates)
    else:
        live = np.ones((n, D + 1), dtype=bool)
        gate_live = [np.ones(len(g), dtype=bool) for g in gates]

    shown = np.arange(n)
    D_draw = D
    if compact:
        if highlight is None:
            raise ValueError("compact=True needs highlight=")
        shown = np.flatnonzero(live.any(axis=1))
        D_draw = int(circuit.out_depths[highlight].max())
        gates = [[g for g, keep in zip(layer, gl) if keep]
                 for layer, gl in zip(gates, gate_live)]
        gate_live = [np.ones(len(g), dtype=bool) for g in gates]
    shown = _wire_order(circuit, shown, order)
    ypos = np.full(n, np.nan)
    ypos[shown] = np.arange(len(shown))[::-1]  # first wire on top
    if labels is None:
        labels = len(shown) <= 40

    # x layout: pack gates into non-overlapping columns within each layer
    bounds, gate_x, gate_wires = [0.0], [], []
    for layer in gates[:D_draw]:
        spans = [(ypos[w].min(), ypos[w].max()) for w, _ in layer]
        col_end = []
        cols = np.zeros(len(layer), dtype=int)
        for j in sorted(range(len(layer)), key=lambda j: spans[j][0]):
            lo, hi = spans[j]
            c = next((c for c, e in enumerate(col_end) if e < lo), len(col_end))
            if c == len(col_end):
                col_end.append(hi)
            col_end[c] = hi
            cols[j] = c
        width = max(len(col_end), 1) * col_width + 0.5
        xs = bounds[-1] + 0.25 + (cols + 0.5) * col_width
        gate_x.append(xs)
        gate_wires.append({int(w_): x for (w, _), x in zip(layer, xs) for w_ in w})
        bounds.append(bounds[-1] + width)
    bounds = np.array(bounds)
    x_end = bounds[-1] + 0.3

    if ax is None:
        _, ax = plt.subplots(figsize=(min(max(2.5 + 0.55 * x_end, 6), 30),
                                      min(max(1.2 + 0.3 * len(shown), 2.5), 30)))
    cm = plt.get_cmap(cmap)
    depth_color = lambda d: cm(0.1 + 0.8 * (d - 1) / max(D - 1, 1))

    # wires: split each layer's span at the gate (value before / after it)
    segs = {True: [], False: [], "frozen": []}
    for w in shown:
        y, d = ypos[w], circuit.out_depths[w]
        for t in range(1, D_draw + 1):
            x0, x1 = bounds[t - 1], bounds[t]
            if t > d:
                segs["frozen"].append([(x0, y), (x1, y)])
            elif w in gate_wires[t - 1]:
                xg = gate_wires[t - 1][w]
                segs[bool(live[w, t - 1])].append([(x0, y), (xg, y)])
                segs[bool(live[w, t])].append([(xg, y), (x1, y)])
            else:
                segs[bool(live[w, t - 1])].append([(x0, y), (x1, y)])
        segs["frozen"].append([(bounds[-1], y), (x_end, y)])
    hl = highlight is not None
    ax.add_collection(LineCollection(segs["frozen"], colors=FADED, lw=0.8,
                                     linestyles=(0, (1, 2)), zorder=1))
    ax.add_collection(LineCollection(segs[False], colors=FADED, lw=0.9, zorder=1))
    ax.add_collection(LineCollection(segs[True], colors=INK, lw=1.4 if hl else 1.0,
                                     zorder=2))

    # gates
    for t, (layer, xs, gl) in enumerate(zip(gates[:D_draw], gate_x, gate_live)):
        for (w, tab), x, keep in zip(layer, xs, gl):
            col = (CONE if hl else INK) if keep else FADED
            ys = ypos[w]
            ax.plot([x, x], [ys.min(), ys.max()], color=col, lw=1.6 if keep else 1.0,
                    zorder=3 if keep else 1.5, solid_capstyle="round")
            ax.scatter(np.full(3, x), ys, s=22, color=col, zorder=4 if keep else 1.6,
                       edgecolors="none")
            if show_tables:
                ax.text(x, ys.max() + 0.22, "".join(map(str, tab)), ha="center",
                        va="bottom", fontsize=6, family="monospace",
                        color=col if keep else "#999999", zorder=5)

    # taps
    for w in shown:
        d = circuit.out_depths[w]
        if d > D_draw:
            continue
        big = hl and w in highlight
        ax.scatter(bounds[d], ypos[w], marker="D" if big else "o",
                   s=70 if big else 26, color=depth_color(d), zorder=6,
                   edgecolors=CONE if big else "white", linewidths=1.4 if big else 0.6)

    if labels:
        for w in shown:
            in_cone = live[w, 0] if hl else True
            ax.text(-0.15, ypos[w], f"$x_{{{w}}}$", ha="right", va="center",
                    fontsize=8, color=INK if in_cone else "#aaaaaa")
            d = circuit.out_depths[w]
            if d <= D_draw:
                ax.text(x_end + 0.1, ypos[w], f"$y_{{{w}}}$ @{d}", ha="left",
                        va="center", fontsize=8,
                        color=CONE if hl and w in highlight else INK)

    for t in range(1, D_draw + 1):
        ax.text((bounds[t - 1] + bounds[t]) / 2, len(shown) - 0.3, str(t),
                ha="center", va="bottom", fontsize=7, color="#888888")
    ax.set_xlim(-0.9 if labels else -0.3, x_end + (1.4 if labels else 0.3))
    ax.set_ylim(-0.8, len(shown) + 0.1)
    ax.axis("off")
    if hl:
        k = int(live[:, 0].sum())
        who = ", ".join(map(str, highlight)) if len(highlight) <= 6 else f"{len(highlight)} wires"
        ax.set_title(f"light cone of y[{who}]: {k} cone inputs, "
                     f"{sum(int(g.sum()) for g in gate_live)} gates", fontsize=10)
    return ax


def plot_function(circuit: Circuit, wire: int, axes=None):
    """Truth table of one output as a Karnaugh map over its relevant inputs
    (Gray-coded rows/columns, so adjacent cells differ in one bit), plus its
    Walsh-Fourier weight by degree. Degree-k weight near 1 = parity-like."""
    inputs, table = restricted_truth_table(circuit, wire)
    k = len(inputs)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 3.8),
                               gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax, ax2 = axes
    kr, kc = k // 2, k - k // 2
    gray = lambda m: np.array([i ^ (i >> 1) for i in range(2 ** m)], dtype=int)
    rows, cols = gray(kr), gray(kc)
    grid = table.reshape(2 ** kr, 2 ** kc)[np.ix_(rows, cols)]
    ax.imshow(grid, cmap="Greys", vmin=-0.3, vmax=1.15, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if 2 ** kc <= 16:
        ax.set_xticks(range(2 ** kc), [format(c, f"0{kc}b") for c in cols],
                      rotation=90, fontsize=7, family="monospace")
    if 0 < 2 ** kr <= 16:
        ax.set_yticks(range(2 ** kr), [format(r, f"0{kr}b") for r in rows],
                      fontsize=7, family="monospace")
    ax.set_ylabel("x[" + ",".join(map(str, inputs[:kr])) + "]", fontsize=8)
    ax.set_xlabel("x[" + ",".join(map(str, inputs[kr:])) + "]", fontsize=8)
    ax.set_title(f"y{wire} (tap depth {circuit.out_depths[wire]}): "
                 f"{k} relevant / {len(cone_inputs(circuit, wire))} cone inputs, "
                 f"mean {table.mean():.2f}", fontsize=9)

    w = walsh_by_degree(table)
    ax2.bar(np.arange(k + 1), w, color=CONE)
    ax2.set(xlabel="degree |S|", ylabel="Fourier weight", ylim=(0, 1),
            xticks=np.arange(k + 1))
    ax2.set_title("Walsh spectrum by degree", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    return axes


def plot_influence(circuit: Circuit, inf=None, ax=None, n_samples=1024):
    """Output x input influence matrix, outputs sorted by tap depth (the
    light cones fanning out as depth grows)."""
    inf = influence(circuit, n_samples) if inf is None else inf
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    o = np.argsort(circuit.out_depths, kind="stable")
    im = ax.imshow(inf[o], aspect="auto", cmap="magma_r", vmin=0,
                   vmax=max(0.5, inf.max()), interpolation="nearest")
    d = circuit.out_depths[o]
    for b in np.flatnonzero(np.diff(d)) + 0.5:
        ax.axhline(b, color="#4a90d9", lw=0.5)
    ks = np.unique(d)
    ks = ks[::max(1, len(ks) // 12)]  # thin labels when there are many depths
    mids = [np.flatnonzero(d == k).mean() for k in ks]
    ax.set_yticks(mids, [f"d{k}" for k in ks], fontsize=7)
    ax.set(xlabel="input wire", ylabel="outputs (grouped by tap depth)",
           title="influence  P[y_o flips | x_i flips]")
    plt.colorbar(im, ax=ax, fraction=0.04)
    return ax


def plot_summary(circuit: Circuit, n_samples: int = 1024, inf=None, fig=None):
    """Rich-task dashboard: depth histogram, support size vs depth,
    total influence vs depth, and the influence matrix."""
    inf = influence(circuit, n_samples) if inf is None else inf
    sup = supports(circuit).sum(axis=1)
    rel = (inf > 0).sum(axis=1)  # sampled: a lower bound on true support
    sens = inf.sum(axis=1)
    d = circuit.out_depths
    dv = np.arange(1, circuit.depth + 1)
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(d))

    fig = fig or plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35])
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax3 = fig.add_subplot(gs[1, :])

    ax0.bar(dv, np.bincount(d, minlength=circuit.depth + 1)[1:], color="#4a90d9")
    ax0.set(xlabel="tap depth", ylabel="# outputs", title="tap depths")

    ax1.scatter(d + jit, sup, s=10, color="#999999", label="structural (light cone)")
    ax1.scatter(d + jit, rel, s=10, color=CONE, label="functional (influence > 0)")
    ax1.plot(dv, np.minimum(3.0 ** dv, circuit.n_wires), "k:", lw=1, label="min(3^d, n)")
    ax1.set(xlabel="tap depth", ylabel="# inputs", yscale="log",
            title="support size per output")
    ax1.legend(fontsize=7)

    ax2.scatter(d + jit, sens, s=10, color=CONE)
    ax2.axhline(circuit.n_wires / 2, color="k", ls=":", lw=1)
    ax2.text(dv[0], circuit.n_wires / 2, " n/2 (random function)", fontsize=7,
             va="bottom")
    ax2.set(xlabel="tap depth", ylabel="total influence", yscale="log",
            title="sensitivity per output")
    for a in (ax0, ax1, ax2):
        a.spines[["top", "right"]].set_visible(False)

    plot_influence(circuit, inf, ax=ax3)
    fig.suptitle(f"brickwork circuit: {circuit.n_wires} wires, depth {circuit.depth}, "
                 f"{sum(map(len, layer_gates(circuit)))} gates", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# tree
# --------------------------------------------------------------------------

def draw_tree(tree: TreeCircuit, tapped=None, subtree=None, ax=None,
              cmap="viridis"):
    """Fan-in-3 tree, leaves at the bottom. Tapped nodes are filled and
    colored by supervision gap (`tree_circuit.supervision_gaps`), untapped
    nodes are hollow.

    tapped   bool (n_interior,) in evaluate_np order (level 1 first);
             default: every node tapped.
    subtree  (level, index): draw only that node's subtree.
    """
    tapped = (np.ones(tree.n_interior, dtype=bool) if tapped is None
              else np.asarray(tapped, dtype=bool))
    gaps = supervision_gaps(tree, tapped)
    offsets = np.concatenate([[0], np.cumsum(tree.level_sizes)])
    top, root = (tree.depth, 0) if subtree is None else subtree
    leaf_lo = root * 3 ** top
    n_leaf = 3 ** top

    if ax is None:
        _, ax = plt.subplots(figsize=(min(2 + 0.12 * n_leaf, 18), 1 + 0.7 * top))
    cm = plt.get_cmap(cmap)
    gmax = max(int(gaps.max()), 1)
    edges, pts = [], []
    for L in range(1, top + 1):
        span = 3 ** (top - L)                   # nodes at level L in the subtree
        idx = root * span + np.arange(span)
        x = (idx * 3 ** L + (3 ** L - 1) / 2) - leaf_lo
        cx = lambda j: (j * 3 ** (L - 1) + (3 ** (L - 1) - 1) / 2) - leaf_lo
        for j, xj in zip(idx, x):
            for c in range(3):
                edges.append([(xj, L), (cx(3 * j + c), L - 1)])
            g = offsets[L - 1] + j
            pts.append((xj, L, tapped[g], gaps[g]))
    ax.add_collection(LineCollection(edges, colors="#b0b0b0",
                                     lw=0.9 if n_leaf <= 243 else 0.3, zorder=1))
    pts = np.array(pts, dtype=float)
    s = 60 if n_leaf <= 81 else 18 if n_leaf <= 729 else 5
    on = pts[:, 2] > 0
    sc = ax.scatter(pts[on, 0], pts[on, 1], c=pts[on, 3], cmap=cm, vmin=1,
                    vmax=gmax, s=s if n_leaf <= 81 else 3 * s, zorder=3, edgecolors="none")
    ax.scatter(pts[~on, 0], pts[~on, 1], s=s, facecolors="white",
               edgecolors="#888888", lw=0.7, zorder=2)
    ax.scatter(np.arange(n_leaf), np.zeros(n_leaf), s=max(s // 4, 1), color=INK,
               marker="s" if n_leaf <= 81 else ".", zorder=2, edgecolors="none")
    ax.set_xlim(-1, n_leaf)
    ax.set_ylim(-0.4, top + 0.4)
    ax.set_yticks(range(top + 1), ["leaves"] + [f"L{L}" for L in range(1, top + 1)])
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    plt.colorbar(sc, ax=ax, fraction=0.03, label="supervision gap")
    where = "" if subtree is None else f", subtree at L{top}[{root}]"
    ax.set_title(f"tree: {tree.n_leaves} leaves, depth {tree.depth}{where}; "
                 f"{int(tapped.sum())}/{tree.n_interior} nodes tapped", fontsize=10)
    return ax
