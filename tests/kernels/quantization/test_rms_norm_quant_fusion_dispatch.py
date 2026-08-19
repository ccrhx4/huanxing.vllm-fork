# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the manual RMSNorm + input-quant fusion producer
(`rms_norm_input_quant`, RFC #43224 / issue #43500).

The fused FP8 static-per-tensor path must produce the same quantized
activation (data and scale) as the unfused reference: RMSNorm (with optional
residual add) followed by static QuantFP8. This covers both the standard
``RMSNorm`` and the Gemma-architecture ``GemmaRMSNorm`` (e.g. Qwen3.5), which
folds a ``(1 + weight)`` offset and must route to the dedicated ``gemma_*``
fused kernels. The dispatcher must also fall back to a plain norm when the
downstream linear advertises no consumable key (or is None), preserving exact
pre-fusion behavior.

Runs on XPU where the fused gemma quant kernels are available.
"""

import pytest
import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.fusion.rms_norm_quant import rms_norm_input_quant
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()
DTYPES = [torch.bfloat16]
NUM_TOKENS = [7, 256]
HIDDEN_SIZES = [64, 2048]
NORM_CLASSES = [RMSNorm, GemmaRMSNorm]

xpu_only = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="fused gemma quant kernels are XPU-only"
)


class _FakeLinear(torch.nn.Module):
    """Carries just the attributes the dispatcher reads."""

    def __init__(self, input_quant_key=None, input_scale=None):
        super().__init__()
        if input_quant_key is not None:
            self.input_quant_key = input_quant_key
        if input_scale is not None:
            self.input_scale = input_scale


def _make_norm(norm_cls, hidden_size, dtype, device):
    norm = norm_cls(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    # RMSNorm weights center on 1.0; GemmaRMSNorm folds +1 so weights are
    # zero-centered small deviations.
    mean = 0.0 if norm_cls is GemmaRMSNorm else 1.0
    norm.weight.data.normal_(mean=mean, std=0.1)
    return norm


@pytest.mark.parametrize("norm_cls", NORM_CLASSES)
def test_no_quant_key_falls_back_to_plain_norm(default_vllm_config, norm_cls):
    """Without an input_quant_key the dispatcher must behave exactly like the
    pre-fusion code path (plain RMSNorm, optional fused-add)."""
    torch.manual_seed(0)
    norm = norm_cls(32, eps=1e-6)
    linear = _FakeLinear()
    x = torch.randn(4, 32)

    out, residual = rms_norm_input_quant(norm, x.clone(), None, linear)
    assert not isinstance(out, QuantizedActivation)
    torch.testing.assert_close(out, norm(x))
    torch.testing.assert_close(residual, x)


@pytest.mark.parametrize("norm_cls", NORM_CLASSES)
def test_linear_none_falls_back_to_plain_norm(default_vllm_config, norm_cls):
    """`linear=None` (decoder-layer subclasses that swap self_attn/mlp for
    modules without the expected projection, e.g. MoE blocks) must take the
    plain-norm path, with and without residual."""
    torch.manual_seed(0)
    norm = norm_cls(32, eps=1e-6)
    x = torch.randn(4, 32)

    out, residual = rms_norm_input_quant(norm, x.clone(), None, None)
    assert not isinstance(out, QuantizedActivation)
    torch.testing.assert_close(out, norm(x))
    torch.testing.assert_close(residual, x)

    res_in = torch.randn(4, 32)
    ref_out, ref_res = norm(x.clone(), res_in.clone())
    out2, res2 = rms_norm_input_quant(norm, x.clone(), res_in.clone(), None)
    assert not isinstance(out2, QuantizedActivation)
    torch.testing.assert_close(out2, ref_out)
    torch.testing.assert_close(res2, ref_res)


@xpu_only
@pytest.mark.parametrize("norm_cls", NORM_CLASSES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_static_per_tensor_matches_unfused(
    default_vllm_config,
    norm_cls,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    with_residual: bool,
):
    torch.manual_seed(0)
    device = "xpu"
    norm = _make_norm(norm_cls, hidden_size, dtype, device)
    scale = torch.tensor(0.05, dtype=torch.float32, device=device)
    linear = _FakeLinear(input_quant_key=kFp8StaticTensorSym, input_scale=scale)

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual: torch.Tensor | None = torch.randn_like(x) if with_residual else None

    # Unfused reference: norm (+ residual add) then static QuantFP8.
    if residual is not None:
        ref_normed, ref_residual = norm(x.clone(), residual.clone())
    else:
        ref_normed, ref_residual = norm(x.clone()), x
    quant_fp8 = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    ref_q, _ = quant_fp8(ref_normed, scale=scale)

    out, out_residual = rms_norm_input_quant(
        norm,
        x.clone(),
        residual.clone() if residual is not None else None,
        linear,
    )

    assert isinstance(out, QuantizedActivation)
    assert out.quant_key == kFp8StaticTensorSym
    assert out.data.dtype == FP8_DTYPE
    assert out.scale is scale
    torch.testing.assert_close(out_residual, ref_residual)
    # The fused kernel keeps the normalized activation in fp32 before the fp8
    # cast, while the unfused reference rounds it to the norm dtype first. At
    # fp8-bucket boundaries this makes a rare element land in an adjacent fp8
    # code (a single-ulp flip). The kernel numerics themselves are validated
    # bit-exactly in the vllm-xpu-kernels test suite; here we only require that
    # such boundary flips are rare and never exceed one fp8 bucket.
    out_f = out.data.to(torch.float32)
    ref_f = ref_q.to(torch.float32)
    close = torch.isclose(out_f, ref_f, atol=0.06, rtol=0.1)
    mismatch_frac = (~close).float().mean().item()
    assert mismatch_frac <= 0.02, (
        f"fp8 quant mismatch fraction too high: {mismatch_frac:.4f}"
    )
    if (~close).any():
        # Mismatches must be single fp8-bucket flips: within ~1/8 (e4m3
        # mantissa resolution) of the reference magnitude, i.e. relative <= 0.2.
        bad = ~close
        rel = (out_f[bad] - ref_f[bad]).abs() / ref_f[bad].abs().clamp_min(1.0)
        assert torch.all(rel <= 0.2), (
            f"fp8 mismatch exceeds one bucket: max rel {rel.max().item():.3f}"
        )


@xpu_only
@pytest.mark.parametrize("norm_cls", NORM_CLASSES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("with_residual", [False, True])
@torch.inference_mode()
def test_dynamic_per_block_matches_unfused(
    default_vllm_config,
    norm_cls,
    num_tokens: int,
    group_size: int,
    with_residual: bool,
):
    """Dynamic per-token-group (block) fp8 path: the fused kernel must match
    the unfused reference (norm + per-group QuantFP8) in the dequantized
    domain, with per-group scales produced at runtime."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic64Sym,
        kFp8Dynamic128Sym,
    )

    torch.manual_seed(0)
    device = "xpu"
    hidden_size = 1024
    key = kFp8Dynamic128Sym if group_size == 128 else kFp8Dynamic64Sym
    norm = _make_norm(norm_cls, hidden_size, torch.bfloat16, device)
    linear = _FakeLinear(input_quant_key=key)

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    residual: torch.Tensor | None = torch.randn_like(x) if with_residual else None

    # fp32 reference matching the kernel's internal math (residual add rounded
    # to bf16, variance + normalization in fp32), then per-group QuantFP8 on the
    # fp32 normed activation. Using norm(x) directly would round to bf16 before
    # quant and shift the per-group scales relative to the fused kernel.
    x32 = x.float()
    if residual is not None:
        z = (x + residual).to(x.dtype)
        ref_residual: torch.Tensor = z
        x32 = z.float()
    else:
        ref_residual = x
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + norm.variance_epsilon)
    w = norm.weight.float() + (1.0 if norm_cls is GemmaRMSNorm else 0.0)
    ref_normed = x32 * inv_rms * w
    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape(1, group_size))
    ref_q, ref_scales = quant_fp8(ref_normed)

    out, out_residual = rms_norm_input_quant(
        norm,
        x.clone(),
        residual.clone() if residual is not None else None,
        linear,
    )

    assert isinstance(out, QuantizedActivation)
    assert out.quant_key == key
    assert out.data.dtype == FP8_DTYPE
    num_groups = hidden_size // group_size
    assert out.scale.shape == (num_tokens, num_groups)
    torch.testing.assert_close(out_residual, ref_residual)

    # Per-group scales derive from the same fp32 normed activation, so they
    # match closely; compare the dequantized activation, tolerating rare
    # single fp8-bucket rounding differences.
    torch.testing.assert_close(
        out.scale.to(torch.float32),
        ref_scales.to(torch.float32).view(num_tokens, num_groups),
        atol=1e-4,
        rtol=1e-4,
    )
    out_deq = out.data.to(torch.float32).view(
        num_tokens, num_groups, group_size
    ) * out.scale.to(torch.float32).unsqueeze(-1)
    ref_deq = ref_q.to(torch.float32).view(
        num_tokens, num_groups, group_size
    ) * ref_scales.to(torch.float32).view(num_tokens, num_groups).unsqueeze(-1)
    torch.testing.assert_close(out_deq, ref_deq, atol=0.2, rtol=0.15)
