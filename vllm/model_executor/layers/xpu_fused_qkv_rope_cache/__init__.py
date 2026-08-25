# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU port of the AITER gated-QKV + QK-RMSNorm + RoPE + KV-cache fusion.

This is the vLLM-side plumbing for the fused kernel investigated in
vllm-project/vllm#52901.  Upstream the kernel ships inside the AMD-only
``aiter`` package and both the compile pass and the per-layer support checks
are hard-gated on ``rocm_aiter_ops.is_enabled()``.  The kernel body itself is
plain ``triton.language`` with nothing AMD-specific in it, so it is vendored
here (MIT, see ``kernel_v1.py`` / ``kernel_v2.py``) and launched on XPU.

Two things differ from upstream:

* ``kernel_v2`` uses FlashInfer's *flattened work axis* -- ``grid.y`` enumerates
  ``QH`` Q(+gate) units, then ``KVH`` K units, then ``KVH`` V units -- instead of
  AITER's ``grid.y = QH`` with an ``if hq < KVH`` branch that makes ``KVH`` of the
  blocks do ~2.5x the work of the rest.  Measured on Arc Pro B60 this is
  performance-neutral (see REPORT.md); it is kept because it is bit-exact,
  never slower on average, and structurally simpler.
* ``BLOCK_T`` comes from a byte-derived rule (a fixed ~4 KiB tile) rather than
  from the AMD CU count, which is worth up to 3.6x on XPU at prefill.

Scope B: the KV-cache write happens *inside* this op.  The caller is
responsible for suppressing the separate ``unified_kv_cache_update`` for the
same layer (see ``disable_separate_kv_cache_update``) so the cache is written
exactly once.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import direct_register_custom_op

from .wrapper import fused_qkv_split_qk_norm_rope_cache

logger = init_logger(__name__)

__all__ = [
    "fused_qkv_split_qk_norm_rope_cache",
    "xpu_fusion_supported",
    "disable_separate_kv_cache_update",
    "fused_qkv_norm_rope_kvcache_gated",
]

# head_dim must be a power of two for the Triton kernel's tl.arange tiles.
_SUPPORTED_HEAD_DIMS = (64, 128, 256)


def xpu_fusion_supported(head_dim: int, is_neox: bool, kv_cache_dtype: str) -> bool:
    """Cheap, side-effect-free eligibility check used to gate the fast path."""
    from vllm.platforms import current_platform

    return (
        current_platform.is_xpu()
        and is_neox
        and head_dim in _SUPPORTED_HEAD_DIMS
        and kv_cache_dtype in ("auto", "bfloat16", "float16")
    )


def disable_separate_kv_cache_update(attn_layer) -> None:
    """Make this layer's standalone KV-cache-update op a no-op.

    Scope B writes the cache from inside the fused op, so the separate
    ``unified_kv_cache_update`` must not write it a second time.  We neutralize
    it *per layer* rather than flipping the backend-wide
    ``forward_includes_kv_cache_update`` class attribute, because that attribute
    also controls whether ``slot_mapping`` is padded to the CUDA-graph batch
    size (``gpu_model_runner``); changing it would silently desynchronize
    ``slot_mapping`` from the token count this kernel indexes with.
    """
    impl = attn_layer.impl
    if getattr(impl, "_xpu_kv_update_disabled", False):
        return

    def _noop(*args, **kwargs):
        return None

    impl.do_kv_cache_update = _noop
    impl._xpu_kv_update_disabled = True


def _split_triton_attn_kv_cache(kv_cache: torch.Tensor, head_size: int):
    """(num_blocks, H, block_size, 2*hs) -> two (num_blocks, block_size, H, hs).

    Same expression TritonAttentionImpl.do_kv_cache_update uses.  The result is
    a pair of *strided* views; the vendored kernel takes fully general cache
    strides, so no contiguous copy is needed.
    """
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    return key_cache, value_cache


def _run_fused_qkv_norm_rope_kvcache(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    gate_out: torch.Tensor | None,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    layer_name: str,
) -> None:
    _, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)

    half = rotary_dim // 2
    cos = cos_sin_cache[..., :half]
    sin = cos_sin_cache[..., half : 2 * half]

    T = qkv.shape[0]
    write_cache = (
        slot_mapping is not None
        and kv_cache is not None
        and kv_cache.dim() == 4
        and kv_cache.numel() > 0
    )
    if os.environ.get("VLLM_XPU_FUSED_DEBUG") == "1":
        logger.info(
            "xpu_fused[%s]: T=%d qkv=%s pos=%s ndim=%s kv_cache=%s slot=%s "
            "write_cache=%s cos_sin=%s",
            layer_name, T, tuple(qkv.shape), tuple(positions.shape),
            positions.ndim,
            None if kv_cache is None else tuple(kv_cache.shape),
            None if slot_mapping is None else tuple(slot_mapping.shape),
            write_cache, tuple(cos_sin_cache.shape),
        )

    if write_cache:
        key_cache, value_cache = _split_triton_attn_kv_cache(kv_cache, head_dim)
        slots = slot_mapping[:T]
    else:
        # Memory-profiling / dummy run: no cache is allocated yet.  Feed a
        # 1-block dummy and an all-invalid slot map so the kernel's existing
        # `slots >= 0` guard skips every cache store, keeping one code path.
        key_cache = qkv.new_zeros((1, 16, num_kv_heads, head_dim))
        value_cache = qkv.new_zeros((1, 16, num_kv_heads, head_dim))
        slots = torch.full((T,), -1, dtype=torch.long, device=qkv.device)

    fused_qkv_split_qk_norm_rope_cache(
        qkv,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        key_cache,
        value_cache,
        slots,
        num_heads,
        num_kv_heads,
        head_dim,
        is_neox=True,
        reuse_freqs_front_part=True,
        attn_output_gate=gate_out is not None,
        eps=eps,
        kv_cache_layout="NHD",
        out={"q": q_out, "k": k_out, "v": v_out, "gate": gate_out},
    )


def fused_qkv_norm_rope_kvcache_gated(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    gate_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    layer_name: str,
) -> None:
    _run_fused_qkv_norm_rope_kvcache(
        q_out, k_out, v_out, gate_out, qkv, positions, q_weight, k_weight,
        cos_sin_cache, num_heads, num_kv_heads, head_dim, rotary_dim, eps,
        layer_name,
    )


def fused_qkv_norm_rope_kvcache(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    layer_name: str,
) -> None:
    """Un-gated variant, for models whose qkv_proj has no attention gate
    (e.g. Qwen3-MoE).  The layout the kernel expects is then simply
    [q | k | v] with no interleaved gate columns."""
    _run_fused_qkv_norm_rope_kvcache(
        q_out, k_out, v_out, None, qkv, positions, q_weight, k_weight,
        cos_sin_cache, num_heads, num_kv_heads, head_dim, rotary_dim, eps,
        layer_name,
    )


def fused_qkv_norm_rope_kvcache_fake(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    layer_name: str,
) -> None:
    return None


def fused_qkv_norm_rope_kvcache_gated_fake(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    gate_out: torch.Tensor,
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="xpu_fused_qkv_norm_rope_kvcache_gated",
    op_func=fused_qkv_norm_rope_kvcache_gated,
    fake_impl=fused_qkv_norm_rope_kvcache_gated_fake,
    mutates_args=["q_out", "k_out", "v_out", "gate_out"],
)

direct_register_custom_op(
    op_name="xpu_fused_qkv_norm_rope_kvcache",
    op_func=fused_qkv_norm_rope_kvcache,
    fake_impl=fused_qkv_norm_rope_kvcache_fake,
    mutates_args=["q_out", "k_out", "v_out"],
)
