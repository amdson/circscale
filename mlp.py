"""Pre-norm residual MLP for learning circuit outputs, in pure JAX.

Architecture (LLaMA-style conventions: bias-free linears, RMSNorm):

    h0 = (2x - 1) @ W_embed                      # bits -> +/-1, then embed
    h_{t+1} = h_t + gelu(rmsnorm(h_t) @ W1_t) @ W2_t   # depth residual blocks
    logits = rmsnorm(h_depth) @ W_head           # (B, n_outputs)

Initialization (all zero-mean normal, biases absent):
  - W_embed: std 1/sqrt(n_inputs). Inputs are +/-1 with unit variance, so the
    stream enters the trunk with unit variance per channel.
  - W1: std sqrt(2/width) (He, matched to the GELU nonlinearity).
  - W2: std 1/sqrt(hidden) * 1/sqrt(depth). The first factor preserves
    variance; the second scales each residual write so the total variance
    added across all `depth` writes is O(1) independent of depth (the GPT-2
    residual-projection scaling, adapted to one write per block).
  - W_head: std 1/sqrt(width), giving O(1) logits at init — near-maximum-
    entropy sigmoid predictions without silencing gradients to the trunk the
    way a zero-init head would.

The residual stream is purely linear: blocks only read through their pre-norm
and write additively.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MLPConfig:
    n_inputs: int = 256
    n_outputs: int = 256
    width: int = 256
    depth: int = 8          # number of residual MLP blocks
    hidden_ratio: int = 4   # block hidden size = hidden_ratio * width


def init_params(key: jax.Array, cfg: MLPConfig) -> dict:
    hidden = cfg.hidden_ratio * cfg.width
    k_embed, k_w1, k_w2, k_head = jax.random.split(key, 4)
    normal = jax.random.normal
    return {
        "embed": normal(k_embed, (cfg.n_inputs, cfg.width)) / jnp.sqrt(cfg.n_inputs),
        "blocks": {
            "norm_scale": jnp.ones((cfg.depth, cfg.width)),
            "w1": normal(k_w1, (cfg.depth, cfg.width, hidden))
            * jnp.sqrt(2.0 / cfg.width),
            "w2": normal(k_w2, (cfg.depth, hidden, cfg.width))
            / jnp.sqrt(hidden * cfg.depth),
        },
        "final_norm_scale": jnp.ones(cfg.width),
        "head": normal(k_head, (cfg.width, cfg.n_outputs)) / jnp.sqrt(cfg.width),
    }


def rmsnorm(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
    return x * scale * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)


def forward(params: dict, x_bits: jax.Array) -> jax.Array:
    """x_bits: (B, n_inputs) in {0,1} -> logits (B, n_outputs)."""
    h = (2.0 * x_bits.astype(jnp.float32) - 1.0) @ params["embed"]

    def block(h, p):
        u = jax.nn.gelu(rmsnorm(h, p["norm_scale"]) @ p["w1"])
        return h + u @ p["w2"], None

    h, _ = jax.lax.scan(block, h, params["blocks"])
    return rmsnorm(h, params["final_norm_scale"]) @ params["head"]


def per_output_bce(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Stable sigmoid BCE, averaged over the batch only -> (n_outputs,).

    Useful for depth-resolved curves: index by circuit.out_depths.
    """
    t = targets.astype(logits.dtype)
    return jnp.mean(jax.nn.softplus(logits) - logits * t, axis=0)


def bce_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean(per_output_bce(logits, targets))
