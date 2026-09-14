# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
    make_unquantized_moe_kernel,
    select_unquantized_moe_backend,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

# Opt-in single-accumulator interleaved SwiGLU GEMM1+activation fusion for
# the XPU unquantized backend, ported from sgl-kernel-xpu (see
# /work/fusemlp/design.md and /work/fusemlp/USAGE.md). Off by default: the
# interleaved gate/up layout only benefits silu/gelu activations, and the
# conversion below replaces `layer.w13_weight`/`w13_bias` in place (freeing
# the pre-interleaved copy) so it is a one-way, sticky change for the layer.
_XPU_FUSED_MOE_INTERLEAVED_ENV = "VLLM_XPU_FUSED_MOE_INTERLEAVED"


def _maybe_interleave_xpu_gate_up_weights(
    layer: "RoutedExperts", moe_config: FusedMoEConfig
) -> bool:
    """Convert `layer.w13_weight`/`w13_bias` to the interleaved layout
    consumed by the fused SwiGLU grouped-GEMM op, in place, when eligible.

    Returns True if the conversion ran (so the caller must skip the regular
    `.transpose(-1, -2)` weight-layout step, since the fused kernel wants
    w13 in its original [E, N, K] layout, not the unfused kernel's
    [E, K, N] layout), and False otherwise.
    """
    if os.environ.get(_XPU_FUSED_MOE_INTERLEAVED_ENV, "0").strip().upper() not in (
        "1",
        "ON",
        "TRUE",
        "YES",
        "Y",
    ):
        return False

    activation = getattr(layer, "activation", None)
    activation_value = getattr(activation, "value", activation)
    if activation_value not in ("silu", "gelu"):
        logger.warning_once(
            "VLLM_XPU_FUSED_MOE_INTERLEAVED is set but activation %s is not "
            "silu/gelu; skipping the interleaved gate/up fusion for this "
            "layer.",
            activation_value,
        )
        return False
    if moe_config.num_local_experts % 8 != 0:
        logger.warning_once(
            "VLLM_XPU_FUSED_MOE_INTERLEAVED is set but num_local_experts=%d "
            "is not a multiple of 8; skipping the interleaved gate/up "
            "fusion for this layer.",
            moe_config.num_local_experts,
        )
        return False

    from vllm_xpu_kernels.moe_utils import interleave_gate_up_weights_xe20

    layer.w13_weight.data = interleave_gate_up_weights_xe20(layer.w13_weight.data)
    layer.w13_weight.xpu_interleaved = True
    if getattr(layer, "w13_bias", None) is not None:
        layer.w13_bias.data = interleave_gate_up_weights_xe20(
            layer.w13_bias.data.to(torch.float32)
        )
    return True




# --8<-- [start:unquantized_fused_moe]
@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    # --8<-- [end:unquantized_fused_moe]

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(
            moe_config=self.moe,
        )

    @property
    def supports_eplb(self) -> bool:
        return True

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory.
        # Skip padding when EPLB is enabled because EPLB requires contiguous
        # weights for the view/rearrangement operations.
        if (
            envs.VLLM_ROCM_MOE_PADDING
            and current_platform.is_rocm()
            and not self.moe.moe_parallel_config.enable_eplb
            and weight.stride(-1) == 1
            and (weight.stride(-2) * weight.element_size()) % 512 == 0
        ):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.accelerator.empty_cache()

        return weight

    def _setup_kernel(
        self,
        layer: "RoutedExperts",
        w13: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        # Shuffle weights to runtime format.
        w13_new, w2_new = convert_to_unquantized_kernel_format(
            self.unquantized_backend,
            moe_config=layer.moe_config,
            w13_weight=w13,
            w2_weight=w2,
        )
        # `moe_kernel` is initialized to None in FusedMoEMethodBase.__init__;
        # On the first call we replace the parameter normally. On subsequent
        # calls (e.g. RL weight updates that re-trigger
        # process_weights_after_loading) the moe kernel has already been set
        # up and CUDA graphs may have captured the parameter addresses, so
        # we copy the shuffled data into the existing storage instead of
        # re-registering a new Parameter.
        is_weight_update = self.moe_kernel is not None  # type: ignore[has-type]
        replace_parameter(layer, "w13_weight", w13_new, prefer_copy=is_weight_update)
        replace_parameter(layer, "w2_weight", w2_new, prefer_copy=is_weight_update)

        if not is_weight_update:
            # Setup moe kernel only on the first call. For the unquantized
            # method, moe_quant_config carries no quantized scales -- only
            # optional w{13,2}_bias references and SwiGLU gate params. Since
            # weight updates mutate those bias tensors in place, the kernel
            # does not need to be re-built.
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            assert self.experts_cls is not None
            self.moe_kernel = make_unquantized_moe_kernel(
                quant_config=self.moe_quant_config,
                moe_config=self.moe,
                backend=self.unquantized_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

            if self.unquantized_backend == UnquantizedMoeBackend.CPU:
                # The CPU experts need the layer itself for the setup that
                # convert_to_unquantized_kernel_format cannot express, since
                # it only sees the two weight tensors: padding and prepacking
                # into the grouped-gemm layout (bias included), and capturing
                # the router config that monolithic apply() cannot carry.
                self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def process_weights_after_loading(self, layer: "RoutedExperts") -> None:
        super().process_weights_after_loading(layer)

        # Padding may allocate a same-shaped strided view. Copy it back instead
        # of rebinding `.data`, since CUDA graphs capture the parameter storage
        # address.
        layer.w13_weight.data.copy_(self._maybe_pad_weight(layer.w13_weight.data))
        layer.w2_weight.data.copy_(self._maybe_pad_weight(layer.w2_weight.data))

        if self.unquantized_backend in [
            UnquantizedMoeBackend.TPU,
            UnquantizedMoeBackend.OOT,
        ]:
            # OOT handles internally.
            return

        elif self.unquantized_backend == UnquantizedMoeBackend.XPU:
            w13 = layer.w13_weight
            w2 = layer.w2_weight

            # The interleaved fusion kernel consumes w13 in its original
            # (pre-transpose) [E, N, K] layout -- see
            # `interleave_gate_up_weights_xe20`'s docstring -- so skip the
            # generic `.transpose(-1, -2)` below when the conversion runs.
            if not _maybe_interleave_xpu_gate_up_weights(layer, self.moe):
                w13.data = w13.transpose(-1, -2).contiguous()
            w2.data = w2.transpose(-1, -2).contiguous()

            self._setup_kernel(
                layer=layer,
                w13=w13,
                w2=w2,
            )
        else:
            self._setup_kernel(
                layer=layer,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
            )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        # SwiGLU/swigluoai gate params live on the layer; plumb them into the
        # quant config so the fused activation (e.g. swigluoai_uninterleave on
        # MiniMax-M3) receives gemm1_clamp_limit/alpha/beta.
        gemm1_alpha = getattr(layer, "swiglu_alpha", None)
        gemm1_beta = getattr(layer, "swiglu_beta", None)
        gemm1_clamp_limit = getattr(layer, "swiglu_limit", None)

        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=gemm1_clamp_limit,
            )

        return FusedMoEQuantConfig.make(
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def forward_native(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def forward_cuda(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward_native(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )

    def apply_monolithic(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
