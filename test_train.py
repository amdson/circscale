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


def test_weight_noise(tmp_path):
    cfg = tiny_cfg(tmp_path, weight_noise=1e-3)
    assert "_wn0.001_" in cfg.name  # transient is the default
    persist = tiny_cfg(tmp_path / "persist", weight_noise=1e-3, noise_mode="persist")
    assert "_wn0.001p_" in persist.name

    # noise is step-derived, so interrupt/resume stays exact
    clean = run(tiny_cfg(tmp_path / "clean", weight_noise=1e-3), quiet=True)
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


def test_adamw_option(tmp_path):
    cfg = tiny_cfg(tmp_path, optimizer="adamw", weight_decay=1e-2)
    assert "_adamwwd0.01_" in cfg.name
    path = run(cfg, quiet=True)
    assert load_run(path)[0]["weight_decay"] == 1e-2
