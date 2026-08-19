# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual RMSNorm + input-quant fusion (RFC #43224, issue #43500).

Producer side of the QuantizedActivation contract for RMSNorm. A decoder
layer asks :func:`rms_norm_input_quant` to normalize its hidden states; when
the downstream linear advertises an ``input_quant_key`` its kernel can consume
(wired by ``expose_input_quant_key``), the norm and the activation
quantization run as a single fused kernel. The result is handed to the linear
as a :class:`QuantizedActivation`, which the scaled-mm kernels read directly
via ``as_quantized_activation``, skipping their own input requantization.

This replaces the ``RMSNormQuantFusionPass`` compiler fusion with an explicit
call site in model code, using the same fused kernels the pass emitted, so
there is no perf delta versus compile-time fusion. Under ``torch.compile`` the
generic ``RMSNormQuantFusionPass`` still handles standard (non-Gemma) norms via
Triton; this eager producer covers execution modes where that pass does not
run (e.g. enforce-eager) and adds a Gemma-aware path the compile pass omits.

Currently wired for FP8 static per-tensor (``kFp8StaticTensorSym``) -- the key
the upstream scaled-mm FP8 kernels (cutlass/flashinfer) advertise. The dynamic
per-token-group (block) keys (``kFp8Dynamic128Sym`` / ``kFp8Dynamic64Sym``,
e.g. Qwen3.5-FP8's 128-block activation quant) route to the fused
``rms_norm_per_block_quant`` kernel, which computes the per-group scales at
runtime. Other keys (dynamic per-tensor/per-token, nvfp4) take the plain-norm
fallback until their producer kernels are added in follow-ups.

Gemma-architecture RMSNorm (``GemmaRMSNorm``, e.g. Qwen3.5) folds a ``(1 +
weight)`` offset applied in fp32; it must route to the dedicated ``gemma_*``
fused kernels rather than the generic ones, which would drop the ``+1`` and be
numerically wrong. When those kernels are unavailable in the installed
backend, the Gemma path falls back to a plain (unfused) norm, preserving
correctness at the cost of the fusion speedup.
"""

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

# Dynamic per-token-group (block) fp8 activation keys handled by the fused
# rms_norm_per_block_quant kernel. Group size is read from the key's scale.
_BLOCK_KEYS = (kFp8Dynamic128Sym, kFp8Dynamic64Sym)


def rms_norm_input_quant(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module | None,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """Apply ``norm`` (with optional residual add) fused with the input
    quantization of ``linear``, when ``linear``'s kernel advertises a key.

    Falls back to the plain (unfused) norm when ``linear`` is ``None`` or has
    no consumable ``input_quant_key``, preserving the exact pre-fusion
    behavior. ``linear=None`` keeps call sites safe for decoder-layer
    subclasses that swap in modules without the expected projection attribute
    (e.g. Aria's MoE layer replacing ``mlp.gate_up_proj``).

    Returns ``(hidden_states, residual)`` like the fused-add RMSNorm path;
    ``hidden_states`` is a :class:`QuantizedActivation` when fusion applied.
    """
    quant_key = getattr(linear, "input_quant_key", None)
    if quant_key == kFp8StaticTensorSym:
        if isinstance(norm, GemmaRMSNorm):
            # Gemma folds (1 + weight) in fp32; only the gemma_* kernels are
            # numerically correct. Fall through to the plain norm when they are
            # not present in the installed backend.
            if hasattr(torch.ops._C, "gemma_rms_norm_static_fp8_quant"):
                return _rms_norm_fp8_static_per_tensor(
                    norm, x, residual, linear, is_gemma=True
                )
        elif hasattr(torch.ops._C, "rms_norm_static_fp8_quant"):
            return _rms_norm_fp8_static_per_tensor(
                norm, x, residual, linear, is_gemma=False
            )
    elif quant_key in _BLOCK_KEYS:
        # Dynamic per-token-group (block) fp8: the scale is computed at runtime
        # by the fused kernel and returned as a [num_tokens, hidden/gs] tensor.
        is_gemma = isinstance(norm, GemmaRMSNorm)
        op_name = (
            "gemma_rms_norm_per_block_quant" if is_gemma else "rms_norm_per_block_quant"
        )
        if hasattr(torch.ops._C, op_name):
            return _rms_norm_fp8_per_block(norm, x, residual, quant_key, is_gemma)

    # No consumable key (or unsupported one) -> plain norm, exact prior path.
    if residual is None:
        return norm(x), x
    out, residual = norm(x, residual)
    return out, residual


def _rms_norm_fp8_static_per_tensor(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    linear: torch.nn.Module,
    is_gemma: bool,
) -> tuple[QuantizedActivation, torch.Tensor]:
    if is_gemma:
        no_residual_op = torch.ops._C.gemma_rms_norm_static_fp8_quant
        residual_op = torch.ops._C.fused_add_gemma_rms_norm_static_fp8_quant
    else:
        no_residual_op = torch.ops._C.rms_norm_static_fp8_quant
        residual_op = torch.ops._C.fused_add_rms_norm_static_fp8_quant

    out_q = torch.empty(x.shape, dtype=current_platform.fp8_dtype(), device=x.device)
    if residual is None:
        no_residual_op(
            out_q,
            x,
            norm.weight.data,
            linear.input_scale,
            norm.variance_epsilon,
        )
        residual = x
    else:
        residual_op(
            out_q,
            x,
            residual,
            norm.weight.data,
            linear.input_scale,
            norm.variance_epsilon,
        )
    return (
        QuantizedActivation(
            data=out_q,
            scale=linear.input_scale,
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=kFp8StaticTensorSym,
        ),
        residual,
    )


def _rms_norm_fp8_per_block(
    norm: torch.nn.Module,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    quant_key,
    is_gemma: bool,
) -> tuple[QuantizedActivation, torch.Tensor]:
    """Fused (optional residual add) RMSNorm + dynamic per-token-group fp8 quant.

    The per-group scale is produced by the kernel (no precomputed input_scale on
    the linear, unlike the static path). ``residual`` is updated in place by the
    kernel when present, matching the fused-add RMSNorm contract.
    """
    from vllm import _custom_ops as ops

    # quant_key.scale.group_shape is (1, gs); the kernel groups along hidden.
    group_shape = quant_key.scale.group_shape
    group_size = [group_shape[0], group_shape[1]]
    out_q, scales = ops.rms_norm_per_block_quant(
        input=x,
        weight=norm.weight.data,
        epsilon=norm.variance_epsilon,
        quant_dtype=current_platform.fp8_dtype(),
        group_size=group_size,
        residual=residual,
        is_gemma=is_gemma,
    )
    # residual is mutated in place when provided; else the input becomes the
    # residual carried to the next fused-add site.
    new_residual = residual if residual is not None else x
    return (
        QuantizedActivation(
            data=out_q,
            scale=scales,
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=quant_key,
        ),
        new_residual,
    )
