import numpy as np

from tree_circuit import (
    evaluate_np,
    make_tree_evaluator,
    sample_tree_circuit,
    supervision_gaps,
)


def test_exact_balance_small():
    # 9 leaves, depth 2: enumerate all 512 inputs; every interior wire is
    # exactly balanced (disjoint supports + balanced gates).
    c = sample_tree_circuit(np.random.default_rng(0), 9, 2)
    x = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(np.uint8)
    y = evaluate_np(c, x)
    assert y.shape == (512, 4)
    assert np.array_equal(y.mean(axis=0), np.full(4, 0.5))


def test_jax_matches_numpy():
    c = sample_tree_circuit(np.random.default_rng(1), 729, 6)
    x = np.random.default_rng(2).integers(0, 2, size=(64, 729), dtype=np.uint8)
    assert np.array_equal(evaluate_np(c, x), np.asarray(make_tree_evaluator(c)(x)))
    assert c.n_interior == 364
    assert np.array_equal(np.bincount(c.out_depths)[1:], [243, 81, 27, 9, 3, 1])


def test_supervision_gaps():
    c = sample_tree_circuit(np.random.default_rng(0), 9, 2)  # 3 level-1 + root
    none = supervision_gaps(c, np.zeros(4, bool))
    assert list(none) == [1, 1, 1, 2]           # untapped: root gap 2
    lvl1 = supervision_gaps(c, np.array([1, 1, 1, 0], bool))
    assert list(lvl1) == [1, 1, 1, 1]           # scaffolded root: gap 1
    partial = supervision_gaps(c, np.array([1, 0, 1, 0], bool))
    assert list(partial) == [1, 1, 1, 2]        # one uncovered child -> gap 2


def test_train_tree_task(tmp_path):
    from train import RunConfig, load_run, run

    cfg = RunConfig(task="tree3", n_wires=27, circ_depth=3, width=32,
                    mlp_depth=2, batch=32, steps=40, warmup=10, eval_every=20,
                    eval_n=64, checkpoint_every=20, out_dir=str(tmp_path))
    assert "_tree3_c27x3" in cfg.name
    _, d = load_run(run(cfg, quiet=True))
    assert d["per_out_loss"].shape[1] == 13
    assert np.array_equal(np.bincount(d["out_depths"])[1:], [9, 3, 1])
