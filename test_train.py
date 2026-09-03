from dataclasses import replace

import numpy as np

from train import RunConfig, load_run, run


def tiny_cfg(out_dir, **kw):
    return RunConfig(
        width=32, mlp_depth=2, batch=32, steps=60, lr=1e-3, warmup=10,
        n_wires=32, circ_depth=8, eval_every=20, eval_n=64,
        checkpoint_every=10, out_dir=str(out_dir), **kw,
    )


def test_run_outputs(tmp_path):
    cfg = tiny_cfg(tmp_path)
    path = run(cfg, quiet=True)
    loaded_cfg, d = load_run(path)
    assert loaded_cfg["width"] == 32
    assert d["train_loss"].shape == (60,) and (d["train_loss"] > 0).all()
    assert list(d["eval_steps"]) == [0, 20, 40, 60]
    assert d["per_out_loss"].shape == (4, 32)
    assert d["param_norm"].shape == (4,) and (d["param_norm"] > 0).all()
    assert not cfg.ckpt_path.exists()

    # idempotent: a second call skips without touching the file
    mtime = path.stat().st_mtime_ns
    assert run(cfg, quiet=True) == path
    assert path.stat().st_mtime_ns == mtime


def test_interrupt_resume_is_exact(tmp_path):
    clean = run(tiny_cfg(tmp_path / "clean"), quiet=True)

    cfg = tiny_cfg(tmp_path / "resumed")
    assert run(cfg, stop_after=25, quiet=True) is None  # interrupted mid-run
    assert cfg.ckpt_path.exists()
    resumed = run(cfg, quiet=True)

    _, a = load_run(clean)
    _, b = load_run(resumed)
    np.testing.assert_allclose(a["train_loss"], b["train_loss"], rtol=1e-6)
    np.testing.assert_allclose(a["per_out_loss"], b["per_out_loss"], rtol=1e-5)
    assert np.array_equal(a["eval_steps"], b["eval_steps"])


def test_output_wires_mask(tmp_path):
    cfg = tiny_cfg(tmp_path, output_wires=(3, 7))
    assert "_ow3-7_" in cfg.name
    path = run(cfg, quiet=True)
    loaded_cfg, d = load_run(path)
    assert loaded_cfg["output_wires"] == [3, 7]
    assert d["per_out_loss"].shape[1] == 32  # eval still covers every output


def test_train_frac(tmp_path):
    cfg = replace(tiny_cfg(tmp_path), n_wires=8, circ_depth=4, train_frac=0.5)
    assert "_tf0.5_" in cfg.name
    _, d = load_run(run(cfg, quiet=True))
    assert d["train_pool"].shape == (128,)
    assert d["per_out_loss_tr"].shape == d["per_out_loss_ho"].shape == (4, 8)
    # full-enumeration eval is the even mix of the two halves
    np.testing.assert_allclose(
        d["per_out_loss"], (d["per_out_loss_tr"] + d["per_out_loss_ho"]) / 2,
        rtol=1e-5,
    )


def test_epoch_order(tmp_path):
    import jax

    from train import _epoch_indices

    # one epoch's batches exactly partition the pool, and epochs reshuffle
    key = jax.random.key(7)
    n_pool, batch = 64, 16
    ep0 = np.concatenate([np.asarray(_epoch_indices(key, s, batch, n_pool))
                          for s in range(4)])
    ep1 = np.concatenate([np.asarray(_epoch_indices(key, s, batch, n_pool))
                          for s in range(4, 8)])
    assert sorted(ep0) == list(range(n_pool)) == sorted(ep1)
    assert not np.array_equal(ep0, ep1)

    cfg = replace(tiny_cfg(tmp_path), n_wires=8, circ_depth=4, train_frac=0.5,
                  batch=32, steps=8, warmup=0, data_order="epoch")
    assert "_tf0.5_epoch_" in cfg.name
    _, d = load_run(run(cfg, quiet=True))  # 8 steps = 2 epochs of 128/32
    assert np.isfinite(d["train_loss"]).all()


def test_train_n(tmp_path):
    cfg = tiny_cfg(tmp_path, train_n=128)  # n_wires=32: not enumerable
    assert "_tn128_" in cfg.name
    _, d = load_run(run(cfg, quiet=True))
    assert d["per_out_loss_tr"].shape == d["per_out_loss_ho"].shape == (4, 32)
    # headline series is the held-out set
    np.testing.assert_allclose(d["per_out_loss"], d["per_out_loss_ho"], rtol=1e-6)
    # pool is memorizable: train loss should sit below held-out by the end
    assert d["per_out_loss_tr"][-1].mean() < d["per_out_loss_ho"][-1].mean()

    epoch = tiny_cfg(tmp_path / "ep", train_n=128, data_order="epoch")
    assert "_tn128_epoch_" in epoch.name
    run(epoch, quiet=True)


def test_weight_noise(tmp_path):
    cfg = tiny_cfg(tmp_path, weight_noise=0.1)
    assert "_wnr0.1_" in cfg.name  # transient + init-relative are the defaults
    persist = tiny_cfg(tmp_path / "persist", weight_noise=0.1, noise_mode="persist")
    assert "_wnr0.1p_" in persist.name
    assert "_wn0.001_" in tiny_cfg(tmp_path, weight_noise=1e-3,
                                   noise_scale="abs").name

    # init_stds mirrors the param tree (norm scales get no noise)
    import jax
    from mlp import MLPConfig, init_params, init_stds
    mcfg = MLPConfig(n_inputs=32, n_outputs=32, width=32, depth=2)
    jax.tree_util.tree_map(lambda p, s: None,
                           init_params(jax.random.key(0), mcfg), init_stds(mcfg))

    # noise is step-derived, so interrupt/resume stays exact
    clean = run(tiny_cfg(tmp_path / "clean", weight_noise=0.1), quiet=True)
    assert run(cfg, stop_after=25, quiet=True) is None
    resumed = run(cfg, quiet=True)
    _, a = load_run(clean)
    _, b = load_run(resumed)
    np.testing.assert_allclose(a["train_loss"], b["train_loss"], rtol=1e-6)

    # both modes actually perturb training, and differently
    _, base = load_run(run(tiny_cfg(tmp_path / "base"), quiet=True))
    _, p = load_run(run(persist, quiet=True))
    assert not np.allclose(a["train_loss"], base["train_loss"])
    assert not np.allclose(p["train_loss"], a["train_loss"])


def test_init_scale_and_adam_eps(tmp_path):
    cfg = tiny_cfg(tmp_path, init_scale=0.3, adam_eps=1e-4)
    assert "_eps0.0001_" in cfg.name and "_is0.3_" in cfg.name
    _, d = load_run(run(cfg, quiet=True))
    _, base = load_run(run(tiny_cfg(tmp_path / "base"), quiet=True))
    # scaled init starts at a smaller norm and trains differently
    assert d["param_norm"][0] < 0.5 * base["param_norm"][0]
    assert not np.allclose(d["train_loss"], base["train_loss"])


def test_sgd_option(tmp_path):
    cfg = replace(tiny_cfg(tmp_path, optimizer="sgd", weight_decay=1e-2), lr=0.1)
    assert "_sgdwd0.01_" in cfg.name
    assert "m0_" in tiny_cfg(tmp_path, optimizer="sgd", momentum=0.0).name
    _, d = load_run(run(cfg, quiet=True))
    assert np.isfinite(d["train_loss"]).all()

    # weight decay actually changes the trajectory
    heavy = replace(tiny_cfg(tmp_path / "wd", optimizer="sgd", weight_decay=0.5), lr=0.1)
    _, h = load_run(run(heavy, quiet=True))
    assert not np.allclose(h["train_loss"], d["train_loss"])


def test_adamw_option(tmp_path):
    cfg = tiny_cfg(tmp_path, optimizer="adamw", weight_decay=1e-2)
    assert "_adamwwd0.01_" in cfg.name
    path = run(cfg, quiet=True)
    assert load_run(path)[0]["weight_decay"] == 1e-2
