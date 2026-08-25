# SPDX-License-Identifier: MIT
# v2 wrapper: FlashInfer-style flattened (GQA-balanced) work axis.
#
# Identical to fused_wrapper.py except that it launches
# _fused_qkv_split_qk_norm_rope_cache_kernel_v2 on a grid of
# (cdiv(T, BLOCK_T), QH + 2*KVH) instead of (cdiv(T, BLOCK_T), QH).
# See fused_kernel_v2.py for the rationale.
# Adapted from ROCm/aiter, commit 072403b37b30c78556f6326af3664679dbdc6367:
#   aiter/ops/triton/rope/fused_qkv_split_qk_norm_rope_cache.py
# Original copyright: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
# All rights reserved. MIT License.
#
# CHANGES vs upstream:
#  1. `infer_rope_cache_triton_block_t` hard-rejects any device.type != "cuda"
#     (ROCm's HIP devices report device.type=="cuda" in PyTorch, so this
#     literally excludes XPU and any other backend even though nothing in the
#     kernel itself is AMD/HIP-specific). We generalize the block size
#     heuristic to also support "xpu" so we can test the *same* kernel body
#     unmodified on Intel GPUs.
#  2. `infer_rope_cache_triton_block_t` now sizes BLOCK_T from the memory
#     geometry (bytes per tile) instead of from T and the CU count. See its
#     docstring. The original CU-based rule is preserved verbatim as
#     `infer_rope_cache_triton_block_t_upstream` for baseline comparison.
# The kernel body itself is unchanged from upstream.

import torch
import triton

import os

# v1 is the default: the FlashInfer-style flattened work axis in v2 was measured
# to be a wash on XPU (1.008x prefill / 1.001x decode, identical n_regs/n_spills)
# and a GQA-ratio control sweep showed the spread is BLOCK_T-plateau noise, not
# imbalance.  v2 is bit-exact with v1, so this switch is perf-only.
_KERNEL_VERSION = os.environ.get("VLLM_XPU_FUSED_KERNEL_VERSION", "v1")
if _KERNEL_VERSION == "v2":
    from .kernel_v2 import (
        _fused_qkv_split_qk_norm_rope_cache_kernel_v2
        as _fused_qkv_split_qk_norm_rope_cache_kernel,
    )
else:
    from .kernel_v1 import _fused_qkv_split_qk_norm_rope_cache_kernel


# Optional in-situ instrumentation: counts real invocations and their shapes
# during an end-to-end vLLM run, so UT-derived predictions can be reconciled
# against what the model actually calls.  Off unless the env var is set.
_INSTRUMENT = os.environ.get("VLLM_XPU_FUSED_INSTRUMENT", "") != ""

# Diagnostic only.  The fusion replaces 7 kernel launches per layer with 1.  In
# decode the device-time saving is ~0.2% but the measured e2e gain is ~3%, so the
# benefit may be host launch/dispatch rather than device work.  Setting this to N
# re-adds N near-zero-cost launches per call, restoring the launch count without
# restoring the device work -- if the gain disappears, launch count is the cause.
_DUMMY_LAUNCHES = int(os.environ.get("VLLM_XPU_FUSED_DUMMY_LAUNCHES", "0"))
_DUMMY_BUF = None

# Diagnostic only: capture buffer addresses on the first call (see wrapper body).
_ALLOC_LOG_ENABLED = os.environ.get("VLLM_XPU_FUSED_ALLOC_LOG", "") != ""
_ALLOC_LOG = None

if _INSTRUMENT:
    import atexit
    import collections
    import json

    _CALL_HIST: dict = collections.Counter()

    def _record_call(T, qh, kvh, head_dim, block_t):
        _CALL_HIST[(T, qh, kvh, head_dim, block_t)] += 1

    def _dump_calls():
        path = os.environ["VLLM_XPU_FUSED_INSTRUMENT"]
        rows = [
            {"T": k[0], "qh": k[1], "kvh": k[2], "head_dim": k[3],
             "block_t": k[4], "count": v}
            for k, v in sorted(_CALL_HIST.items())
        ]
        with open(f"{path}.{os.getpid()}.json", "w") as f:
            json.dump(rows, f, indent=1)

    atexit.register(_dump_calls)


"""BLOCK_T selection.

Two heuristics live here:

``infer_rope_cache_triton_block_t_upstream``
    The original AMD heuristic, ``next_pow2(cdiv(T, 2*cu_count))`` clamped to
    ``[1, 32]``, generalized to accept XPU devices. Kept verbatim so the
    benchmark scripts can still report a truthful "AITER default" baseline
    column. **Not** used by the wrapper any more.

``infer_rope_cache_triton_block_t``
    The fix. Sizes the tile in *bytes* rather than from the core count.

Why the change: the kernel is bandwidth-bound, and measurement across 5
deployment configs x 4 sequence lengths shows the optimum is a two-point
plateau at a 2-4 KiB tile, with steep walls on either side (1.05x at 1 KiB,
1.6x at 8 KiB, 3.06x at 16 KiB, 4.46x at 32 KiB). The plateau sits at the same
*byte* sizes for both head dims measured -- BLOCK_T 4-8 at head_dim=256 and
8-16 at head_dim=128 are both 2048-4096 B. So the right tile size is a property
of the memory system, not of the head count or the core count.

The upstream rule derives BLOCK_T from ``T`` and the CU count, so at large T it
saturates at the 32 clamp = a 16 KiB tile at head_dim=256, which measured 3.06x
slower than the plateau. The byte-derived rule lands on the plateau by
construction and captured 98.0-100.0% (mean 99.6%) of the gain available from
exhaustive per-shape autotuning, worst case 1.016x off the autotuned best.

The formula is the same one FlashInfer uses in
``include/flashinfer/pos_enc.cuh`` (``RopeQuantizeAppendPagedKVCache``):
``vec_size = 32/sizeof(DType); bdx = cdiv(head_dim, vec_size);
bdy = max(128, bdx)/bdx``, i.e. 32 bytes per thread over a 128-thread block.
"""

# Bytes each thread handles; 32 B => a 4096 B tile per 128-thread block.
# 16 (a 2048 B tile) sits on the other plateau point and measured equally well.
_BYTES_PER_THREAD = 32
_NUM_THREADS = 128


def infer_rope_cache_triton_block_t(
    T: int,
    device: torch.device,
    head_dim: int,
    itemsize: int,
    bytes_per_thread: int = _BYTES_PER_THREAD,
    num_threads: int = _NUM_THREADS,
) -> int:
    """Pick Triton token tile ``BLOCK_T`` from the memory-system geometry.

    Returns the number of tokens whose ``head_dim`` rows fill one
    ``num_threads``-wide block at ``bytes_per_thread`` bytes per thread, i.e.
    ``BLOCK_T ~= (num_threads * bytes_per_thread) / (head_dim * itemsize)``.

    Independent of ``T`` and of the device core count, so it needs no device
    query and is portable across CUDA/HIP/XPU by construction. ``device`` is
    accepted for signature compatibility and validation only.
    """
    if device.type not in ("cuda", "xpu"):
        raise ValueError(
            "fused_qkv_split_qk_norm_rope_cache expects a CUDA/HIP or XPU device "
            f"(got {device!r})."
        )
    if head_dim < 1 or itemsize < 1:
        raise ValueError(f"head_dim and itemsize must be >= 1 (got {head_dim}, {itemsize})")

    vec_size = max(1, bytes_per_thread // itemsize)
    bdx = max(1, triton.cdiv(head_dim, vec_size))
    block_t = max(num_threads, bdx) // bdx
    # Never tile more tokens than exist; a partial tile is pure masked waste and
    # this matters in the decode regime where T can be 1.
    block_t = min(block_t, triton.next_power_of_2(max(1, T)))
    return max(1, int(block_t))


def infer_rope_cache_triton_block_t_upstream(T: int, device: torch.device) -> int:
    """The original AMD CU-count heuristic. Retained for baseline comparison.

    Same heuristic as upstream AITER, generalized to also accept XPU devices
    (upstream only accepted device.type == "cuda").
    """
    if device.type == "cuda":
        cu_count = max(
            int(
                torch.cuda.get_device_properties(
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                ).multi_processor_count
            ),
            1,
        )
    elif device.type == "xpu":
        props = torch.xpu.get_device_properties(
            device.index if device.index is not None else torch.xpu.current_device()
        )
        # Intel GPU device properties expose Xe-core / EU-ish "gpu_subslice_count"
        # (analogous role to SM/CU count for grid-size heuristics). Fall back to
        # gpu_eu_count if unavailable so this remains best-effort.
        cu_count = max(
            int(
                getattr(
                    props,
                    "gpu_subslice_count",
                    getattr(props, "gpu_eu_count", 1),
                )
            ),
            1,
        )
    else:
        raise ValueError(
            "fused_qkv_split_qk_norm_rope_cache expects a CUDA/HIP or XPU device "
            f"(got {device!r})."
        )
    block_t = triton.next_power_of_2(triton.cdiv(T, 2 * cu_count))
    return max(1, min(int(block_t), 32))


def fused_qkv_split_qk_norm_rope_cache(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    qh: int,
    kvh: int,
    head_dim: int,
    is_neox: bool = True,
    offsets: torch.Tensor = None,
    reuse_freqs_front_part: bool = True,
    attn_output_gate: bool = False,
    k_scale: torch.Tensor = None,
    v_scale: torch.Tensor = None,
    eps: float = 1e-5,
    gated_qkv_layout: str = "interleaved",
    kv_cache_layout: str = "HND",
    # --- autotuning overrides (XPU-only addition, not present upstream) ---
    # When left as None, BLOCK_T comes from the byte-derived heuristic
    # (see infer_rope_cache_triton_block_t) and num_warps=4/num_stages=default.
    block_t_override: int = None,
    num_warps_override: int = None,
    num_stages_override: int = None,
    out: dict = None,
):
    T = qkv.shape[0]
    q_size = qh * head_dim
    kv_size = kvh * head_dim

    layout = kv_cache_layout.upper()
    if layout not in ("HND", "NHD"):
        raise ValueError(
            'kv_cache_layout must be "HND" or "NHD" ' f"(got {kv_cache_layout!r})."
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            "key_cache and value_cache must have the same shape "
            f"(got {tuple(key_cache.shape)} vs {tuple(value_cache.shape)})."
        )
    num_blocks = key_cache.shape[0]
    if layout == "HND":
        if key_cache.shape[1] != kvh or key_cache.shape[3] != head_dim:
            raise ValueError(
                "HND key_cache expected "
                f"[num_blocks, {kvh}, block_size, {head_dim}], "
                f"got {tuple(key_cache.shape)}."
            )
        block_size = key_cache.shape[2]
        key_cache_stride_t = key_cache.stride(0)
        key_cache_stride_h = key_cache.stride(1)
        key_cache_stride_b = key_cache.stride(2)
        key_cache_stride_d = key_cache.stride(3)
        value_cache_stride_t = value_cache.stride(0)
        value_cache_stride_h = value_cache.stride(1)
        value_cache_stride_b = value_cache.stride(2)
        value_cache_stride_d = value_cache.stride(3)
    else:
        if key_cache.shape[2] != kvh or key_cache.shape[3] != head_dim:
            raise ValueError(
                "NHD key_cache expected "
                f"[num_blocks, block_size, {kvh}, {head_dim}], "
                f"got {tuple(key_cache.shape)}."
            )
        block_size = key_cache.shape[1]
        key_cache_stride_t = key_cache.stride(0)
        key_cache_stride_b = key_cache.stride(1)
        key_cache_stride_h = key_cache.stride(2)
        key_cache_stride_d = key_cache.stride(3)
        value_cache_stride_t = value_cache.stride(0)
        value_cache_stride_b = value_cache.stride(1)
        value_cache_stride_h = value_cache.stride(2)
        value_cache_stride_d = value_cache.stride(3)

    total_num_kv_cache_tokens = num_blocks * block_size

    assert qh >= kvh and qh % kvh == 0, "qh must be multiple of kvh"
    # `out` lets the caller pass pre-allocated destinations, so the enclosing
    # torch.compile custom op can declare them in `mutates_args` (the upstream
    # convention for fused ops with a hidden KV-cache side effect). Writing
    # into caller-owned tensors is what pins this op's position in the graph.
    def _dest(name, shape):
        t = out.get(name) if out else None
        if t is None:
            return torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
        return t.view(shape)

    q = _dest("q", (T, qh, head_dim))
    k = _dest("k", (T, kvh, head_dim))
    v = _dest("v", (T, kvh, head_dim))

    if attn_output_gate:
        gate = _dest("gate", (T, qh, head_dim))
    else:
        gate = None

    global _ALLOC_LOG
    if _ALLOC_LOG_ENABLED and _ALLOC_LOG is None:
        # Record the addresses of the op's buffers on the very first call, so an
        # intermittent per-process performance mode can be correlated with
        # allocation alignment (3 KiB DRAM stride aliasing is the suspect).
        def _al(t):
            return None if t is None else {
                "ptr_mod_3072": t.data_ptr() % 3072,
                "ptr_mod_4096": t.data_ptr() % 4096,
                "stride0_bytes": t.stride(0) * t.element_size(),
            }

        _ALLOC_LOG = {
            "T": T, "qh": qh, "kvh": kvh,
            "qkv_stride_bytes": qkv.stride(0) * qkv.element_size(),
            "qkv": _al(qkv), "q": _al(q), "k": _al(k), "v": _al(v),
            "gate": _al(gate), "preallocated": bool(out),
        }

    if attn_output_gate:
        assert qkv.shape[-1] == 2 * q_size + 2 * kv_size, "Shape error"
        assert gated_qkv_layout in (
            "interleaved",
            "blocked",
        ), 'gated_qkv_layout must be "interleaved" or "blocked"'
    else:
        assert qkv.shape[-1] == q_size + 2 * kv_size, "Shape error"
    assert head_dim == triton.next_power_of_2(head_dim), "head_dim should be power of 2"

    assert cos.shape[-1] == sin.shape[-1], "cos and sin must match in last dim"
    ROTARY_DIM_EFFECTIVE = cos.shape[-1]

    BLOCK_D = head_dim
    BLOCK_D_HALF = head_dim // 2

    BLOCK_T = (
        block_t_override
        if block_t_override is not None
        else infer_rope_cache_triton_block_t(
            T, qkv.device, head_dim, qkv.element_size()
        )
    )
    num_warps = num_warps_override if num_warps_override is not None else 4
    # v1: grid.y is one program per query head, with an `if hq < KVH` branch
    # selecting which programs also do the K/V paths.
    # v2: FlashInfer-style flattened work axis -- grid.y enumerates every
    # homogeneous work unit (QH Q(+gate), then KVH K, then KVH V).
    grid = (
        triton.cdiv(T, BLOCK_T),
        qh + 2 * kvh if _KERNEL_VERSION == "v2" else qh,
    )

    launch_kwargs = dict(num_warps=num_warps)
    if num_stages_override is not None:
        launch_kwargs["num_stages"] = num_stages_override

    if _INSTRUMENT:
        _record_call(T, qh, kvh, head_dim, BLOCK_T)

    if _DUMMY_LAUNCHES:
        global _DUMMY_BUF
        if _DUMMY_BUF is None or _DUMMY_BUF.device != qkv.device:
            _DUMMY_BUF = torch.zeros(1, device=qkv.device, dtype=torch.float32)
        for _ in range(_DUMMY_LAUNCHES):
            _DUMMY_BUF.add_(1.0)

    _fused_qkv_split_qk_norm_rope_cache_kernel[grid](
        qkv_ptr=qkv,
        q_weight_ptr=q_weight,
        k_weight_ptr=k_weight,
        cos_ptr=cos,
        sin_ptr=sin,
        pos_ptr=positions,
        off_ptr=offsets,
        q_ptr=q,
        gate_ptr=gate,
        k_ptr=k,
        v_ptr=v,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        T=T,
        eps=eps,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        stride_qkv_t=qkv.stride(0),
        stride_qkv_d=qkv.stride(1),
        stride_cos_t=cos.stride(0),
        stride_cos_d=cos.stride(-1),
        stride_pos_t=positions.stride(0),
        stride_q_t=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_kv_t=k.stride(0),
        stride_kv_h=k.stride(1),
        stride_kv_d=k.stride(2),
        key_cache_stride_t=key_cache_stride_t,
        key_cache_stride_h=key_cache_stride_h,
        key_cache_stride_d=key_cache_stride_d,
        key_cache_stride_b=key_cache_stride_b,
        value_cache_stride_t=value_cache_stride_t,
        value_cache_stride_h=value_cache_stride_h,
        value_cache_stride_d=value_cache_stride_d,
        value_cache_stride_b=value_cache_stride_b,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        HAVE_POS=(positions is not None),
        HAVE_OFFS=(offsets is not None),
        ENABLE_GATED_Q=attn_output_gate,
        QH=qh,
        KVH=kvh,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        BLOCK_SIZE=block_size,
        ROTARY_DIM_EFFECTIVE=ROTARY_DIM_EFFECTIVE,
        BLOCKED_GATED_LAYOUT=(attn_output_gate and gated_qkv_layout == "blocked"),
        HAVE_K_SCALE=k_scale is not None,
        HAVE_V_SCALE=v_scale is not None,
        total_num_kv_cache_tokens=total_num_kv_cache_tokens,
        **launch_kwargs,
    )

    if attn_output_gate:
        return q, gate, k, v
    else:
        return q, k, v
