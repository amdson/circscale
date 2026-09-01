import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mlp import MLPConfig, bce_loss, forward, init_params
from random_circuit import evaluate_np, sample_circuit


def test_shapes_and_determinism():
    cfg = MLPConfig(n_inputs=32, n_outputs=48, width=64, depth=3)
    params = init_params(jax.random.key(0), cfg)
    x = jax.random.bernoulli(jax.random.key(1), shape=(17, 32)).astype(jnp.uint8)
    logits = forward(params, x)
    assert logits.shape == (17, 48)
    assert jnp.array_equal(logits, forward(params, x))


@pytest.mark.parametrize("width", [64, 512])
@pytest.mark.parametrize("depth", [2, 32])
def test_init_scale(width, depth):
    # Logits are O(1) at init regardless of width and depth.
    cfg = MLPConfig(n_inputs=256, n_outputs=256, width=width, depth=depth)
    params = init_params(jax.random.key(0), cfg)
    x = jax.random.bernoulli(jax.random.key(1), shape=(256, 256)).astype(jnp.uint8)
    std = float(jnp.std(forward(params, x)))
    assert 0.2 < std < 3.0


def test_depth_invariant_stream():
    # The 1/sqrt(depth) residual scaling keeps pre-head stream RMS flat in depth.
    x = jax.random.bernoulli(jax.random.key(1), shape=(256, 64)).astype(jnp.uint8)
    rms = {}
    for depth in (2, 64):
        cfg = MLPConfig(n_inputs=64, n_outputs=8, width=128, depth=depth)
        params = init_params(jax.random.key(0), cfg)
        h = (2.0 * x.astype(jnp.float32) - 1.0) @ params["embed"]

        def block(h, p):
            from mlp import rmsnorm
            return h + jax.nn.gelu(rmsnorm(h, p["norm_scale"]) @ p["w1"]) @ p["w2"], None

        h, _ = jax.lax.scan(block, h, params["blocks"])
        rms[depth] = float(jnp.sqrt(jnp.mean(h * h)))
    assert rms[64] / rms[2] < 1.5


def test_gradients_flow_everywhere():
    cfg = MLPConfig(n_inputs=32, n_outputs=32, width=64, depth=4)
    params = init_params(jax.random.key(0), cfg)
    x = jax.random.bernoulli(jax.random.key(1), shape=(64, 32)).astype(jnp.uint8)
    y = jax.random.bernoulli(jax.random.key(2), shape=(64, 32)).astype(jnp.float32)
    grads = jax.grad(lambda p: bce_loss(forward(p, x), y))(params)
    for path, g in jax.tree_util.tree_flatten_with_path(grads)[0]:
        assert jnp.all(jnp.isfinite(g)), path
        assert float(jnp.max(jnp.abs(g))) > 0, path


def test_memorizes_circuit_batch():
    # Integration: a small model fits a fixed batch of circuit samples.
    rng = np.random.default_rng(0)
    circuit = sample_circuit(rng, n_wires=16, depth=6)
    x = rng.integers(0, 2, size=(64, 16), dtype=np.uint8)
    y = evaluate_np(circuit, x).astype(np.float32)

    cfg = MLPConfig(n_inputs=16, n_outputs=16, width=128, depth=4)
    params = init_params(jax.random.key(0), cfg)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(lambda p: bce_loss(forward(p, x), y))(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = []
    for _ in range(300):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
    assert losses[0] > 0.5          # starts near ln(2) ~ 0.693
    assert losses[-1] < 0.1 * losses[0]
