# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.config
from tests.compile.backend import TestBackend
from tests.kernels.quant_utils import FP8_DTYPE
from tests.kernels.utils import fp8_ulp_distance
from vllm.compilation.passes.fusion.rms_quant_fusion import RMSNormQuantFusionPass
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform


def test_gemma_rms_norm_quant_fusion_fires():
    """Proves RMSNormQuantFusionPass fuses GemmaRMSNorm (weight_bias=1.0)
    calls into the same, unmodified FP8 dynamic per-token quant kernel used
    for plain RMSNorm."""
    dtype = torch.bfloat16
    hidden_size = 256
    num_tokens = 64
    eps = 1e-6

    class GemmaFusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = GemmaRMSNorm(hidden_size, eps)
            self.quant = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

        def forward(self, x):
            x = torch.relu(x)
            y = self.norm(x)
            out, scale = self.quant(y)
            return out, scale

    vllm_config = VllmConfig(
        model_config=ModelConfig(dtype=dtype),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm", "+quant_fp8"],
            pass_config=PassConfig(fuse_norm_quant=True, eliminate_noops=True),
        ),
    )

    with (
        vllm.config.set_current_vllm_config(vllm_config),
        vllm_config.kernel_config.ir_op_priority.set_priority(),
    ):
        torch.set_default_device(current_platform.device_type)
        torch.set_default_dtype(dtype)
        torch.manual_seed(1)

        fusion_pass = RMSNormQuantFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        model = GemmaFusionModel()
        model.norm.weight.data.normal_(mean=0.0, std=0.1)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)
        backend2 = TestBackend(noop_pass, cleanup_pass)

        x = torch.rand(num_tokens, hidden_size)
        torch._dynamo.mark_dynamic(x, 0)

        model_fused = torch.compile(model, backend=backend)
        out_fused, scale_fused = model_fused(x)

        model_unfused = torch.compile(model, backend=backend2)
        out_unfused, scale_unfused = model_unfused(x)

        assert fusion_pass.matched_count > 0, (
            "Gemma RMSNorm+FP8 quant fusion did NOT fire!"
        )

        # FP8 has coarse mantissa precision, so fused vs. unfused paths can
        # land on opposite sides of a rounding tie for a non-trivial fraction
        # of elements; tolerate ulp-1 (adjacent-representable-value) noise,
        # but fail if any element is off by more than that (a real bug would
        # show up as large jumps, not just adjacent-bucket ties).
        assert out_fused.dtype == FP8_DTYPE
        ulp = fp8_ulp_distance(out_fused, out_unfused)
        assert ulp.max().item() <= 1, (
            f"FP8 quant mismatch: max ulp distance {ulp.max().item()} > 1"
        )
        torch.testing.assert_close(scale_fused, scale_unfused, atol=1e-2, rtol=1e-2)
