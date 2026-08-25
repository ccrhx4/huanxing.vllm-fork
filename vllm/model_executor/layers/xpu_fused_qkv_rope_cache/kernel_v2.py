# SPDX-License-Identifier: MIT
# Derived from ROCm/aiter, commit 072403b37b30c78556f6326af3664679dbdc6367:
#   aiter/ops/triton/rope/fused_qkv_split_qk_norm_rope_cache.py
# Original copyright: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
# All rights reserved. MIT License.
"""v2: FlashInfer-style flattened work axis (GQA-balanced).

**The only change vs v1 is the work decomposition.** Every arithmetic helper
(`_rms_norm`, `_rms_norm_returning_inv`, `_partial_neox_rotated`,
`_partial_gptj_rotated`, `_get_neox_rotated_x`, `_get_gptj_rotated_x`) is
imported unmodified from the frozen v1 snapshot, so any measured difference is
attributable to the decomposition and not to drift in the math.

v1 (AITER):

    grid = (cdiv(T, BLOCK_T), QH)
    every program runs the Q path; `if hq < KVH` additionally runs K and V.

    => KVH programs do ~10 tiles of work, QH-KVH programs do ~4.

v2 (this file), mirroring FlashInfer's `RopeQuantizeAppendPagedKVCache`
(`include/flashinfer/pos_enc.cuh:1155` for the grid, `:850-853` for the
cumulative-boundary decode):

    grid = (cdiv(T, BLOCK_T), QH + 2*KVH)

    by in [0,        QH)          -> Q (+ gate) for q head  `by`
    by in [QH,       QH+KVH)      -> K          for kv head `by - QH`
    by in [QH+KVH,   QH+2*KVH)    -> V          for kv head `by - QH - KVH`

    => every program does one homogeneous unit of work; the `if hq < KVH`
       branch is gone.

Secondary effect, also copied from FlashInfer: cos/sin are loaded only by the
segments that actually rotate (Q and K). V blocks skip that load entirely,
which claws back most of the cost of Q and K no longer sharing one load.
"""

import triton
import triton.language as tl

from .kernel_v1 import (
    _partial_gptj_rotated,
    _partial_neox_rotated,
    _rms_norm,
    _rms_norm_returning_inv,
)
from .rope_helpers import _get_gptj_rotated_x, _get_neox_rotated_x


@triton.jit
def _load_cos_sin(
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    t_offs,
    t_mask,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    PARTIAL_ROTATION: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EFFECTIVE_RDH: tl.constexpr,
):
    """Identical to v1's inline cos/sin preamble, factored so that the Q and K
    segments can each call it and the V segment can skip it."""
    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    d_offs = tl.arange(0, BLOCK_D)
    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < EFFECTIVE_RDH),
                d_cos_offs,
                d_cos_offs - EFFECTIVE_RDH,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < EFFECTIVE_RDH
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < EFFECTIVE_RDH
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < EFFECTIVE_RDH * 2

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    if PARTIAL_ROTATION:
        cos = tl.load(cos_ptr + cos_offs, mask=cos_mask, other=1.0)
        sin = tl.load(sin_ptr + cos_offs, mask=cos_mask, other=0.0)
    else:
        cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
        sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)
    return cos, sin


@triton.jit
def _fused_qkv_split_qk_norm_rope_cache_kernel_v2(
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    q_ptr,
    gate_ptr,
    k_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    T,
    eps,
    stride_qkv_t,
    stride_qkv_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_kv_t,
    stride_kv_h,
    stride_kv_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    k_scale_ptr,
    v_scale_ptr,
    total_num_kv_cache_tokens: tl.int64,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    ENABLE_GATED_Q: tl.constexpr,
    QH: tl.constexpr,
    KVH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # PagedAttention block size
    ROTARY_DIM_EFFECTIVE: tl.constexpr = 0,
    BLOCKED_GATED_LAYOUT: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
):
    tl.assume(stride_qkv_t > 0)
    tl.assume(stride_qkv_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_q_t > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_t > 0)
    tl.assume(stride_kv_h > 0)
    tl.assume(stride_kv_d > 0)

    ROTARY_SPAN: tl.constexpr = (
        ROTARY_DIM_EFFECTIVE * 2 if REUSE_FREQS_FRONT_PART else ROTARY_DIM_EFFECTIVE
    )
    PARTIAL_ROTATION: tl.constexpr = ROTARY_SPAN < BLOCK_D
    EFFECTIVE_RDH: tl.constexpr = ROTARY_SPAN // 2

    # --- flattened, segmented work axis (see module docstring) ---
    Q_END: tl.constexpr = QH
    K_END: tl.constexpr = QH + KVH

    if ENABLE_GATED_Q:
        Q_HEAD_STRIDE: tl.constexpr = 2 * BLOCK_D
    else:
        Q_HEAD_STRIDE: tl.constexpr = BLOCK_D
    Q_SIZE: tl.constexpr = QH * Q_HEAD_STRIDE
    KV_SIZE: tl.constexpr = KVH * BLOCK_D

    pid_t = tl.program_id(0)
    by = tl.program_id(1)

    tl.assume(pid_t >= 0)
    tl.assume(by >= 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T
    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        qk_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        qk_rotated_mask = (d_offs % 2 == 0)[None, :]

    if by < Q_END:
        # ================= Q segment (+ gate) =================
        hq = by
        cos, sin = _load_cos_sin(
            cos_ptr, sin_ptr, pos_ptr, off_ptr, t_offs, t_mask,
            stride_cos_t, stride_cos_d, stride_pos_t,
            HAVE_POS, HAVE_OFFS, REUSE_FREQS_FRONT_PART, IS_NEOX,
            PARTIAL_ROTATION, BLOCK_D, EFFECTIVE_RDH,
        )

        if ENABLE_GATED_Q:
            if BLOCKED_GATED_LAYOUT:
                q_lane_base = hq * BLOCK_D
                gate_lane_base = QH * BLOCK_D + hq * BLOCK_D
            else:
                q_lane_base = hq * (2 * BLOCK_D)
                gate_lane_base = hq * (2 * BLOCK_D) + BLOCK_D
        else:
            q_lane_base = hq * BLOCK_D
            gate_lane_base = 0

        q_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + (q_lane_base + d_offs)[None, :] * stride_qkv_d
        )
        q = tl.load(qkv_ptr + q_in_offs, mask=x_mask)

        q_weight = tl.load(q_weight_ptr + d_offs)
        if PARTIAL_ROTATION:
            q, inv_rms_q = _rms_norm_returning_inv(q, q_weight, BLOCK_D, eps)
        else:
            q = _rms_norm(q, q_weight, BLOCK_D, eps)

        if ENABLE_GATED_Q:
            gate_in_offs = (
                t_offs[:, None] * stride_qkv_t
                + (gate_lane_base + d_offs)[None, :] * stride_qkv_d
            )
            gate = tl.load(qkv_ptr + gate_in_offs, mask=x_mask)
            gate_out_offs = (
                t_offs[:, None] * stride_q_t
                + d_offs[None, :] * stride_q_d
                + hq * stride_q_h
            )
            tl.store(gate_ptr + gate_out_offs, gate, mask=x_mask)

        if PARTIAL_ROTATION:
            if IS_NEOX:
                q_rotated = _partial_neox_rotated(
                    inv_rms_q, q_weight_ptr, qkv_ptr, t_offs, d_offs,
                    stride_qkv_t, stride_qkv_d, q_lane_base, x_mask, EFFECTIVE_RDH,
                )
            else:
                q_rotated = _partial_gptj_rotated(
                    inv_rms_q, q_weight_ptr, qkv_ptr, t_offs, d_offs,
                    stride_qkv_t, stride_qkv_d, q_lane_base, x_mask, ROTARY_SPAN,
                )
        elif IS_NEOX:
            q_rotated = _get_neox_rotated_x(
                q, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )
        else:
            q_rotated = _get_gptj_rotated_x(
                q, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )

        q_out_offs = (
            t_offs[:, None] * stride_q_t
            + d_offs[None, :] * stride_q_d
            + hq * stride_q_h
        )
        q = q * cos + q_rotated * sin
        tl.store(q_ptr + q_out_offs, q.to(q_ptr.dtype.element_ty), mask=x_mask)

    elif by < K_END:
        # ================= K segment =================
        hkv = by - Q_END
        cos, sin = _load_cos_sin(
            cos_ptr, sin_ptr, pos_ptr, off_ptr, t_offs, t_mask,
            stride_cos_t, stride_cos_d, stride_pos_t,
            HAVE_POS, HAVE_OFFS, REUSE_FREQS_FRONT_PART, IS_NEOX,
            PARTIAL_ROTATION, BLOCK_D, EFFECTIVE_RDH,
        )

        if HAVE_K_SCALE:
            k_scale = tl.load(k_scale_ptr)
        else:
            k_scale = 1

        KV_HEAD_OFFS = hkv * BLOCK_D
        k_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + KV_HEAD_OFFS) + d_offs)[None, :] * stride_qkv_d
        )
        k = tl.load(qkv_ptr + k_in_offs, mask=x_mask)

        k_weight = tl.load(k_weight_ptr + d_offs)
        if PARTIAL_ROTATION:
            k, inv_rms_k = _rms_norm_returning_inv(k, k_weight, BLOCK_D, eps)
            K_HEAD_D_OFFSET = Q_SIZE + KV_HEAD_OFFS
            if IS_NEOX:
                k_rotated = _partial_neox_rotated(
                    inv_rms_k, k_weight_ptr, qkv_ptr, t_offs, d_offs,
                    stride_qkv_t, stride_qkv_d, K_HEAD_D_OFFSET, x_mask, EFFECTIVE_RDH,
                )
            else:
                k_rotated = _partial_gptj_rotated(
                    inv_rms_k, k_weight_ptr, qkv_ptr, t_offs, d_offs,
                    stride_qkv_t, stride_qkv_d, K_HEAD_D_OFFSET, x_mask, ROTARY_SPAN,
                )
        else:
            k = _rms_norm(k, k_weight, BLOCK_D, eps)
            if IS_NEOX:
                k_rotated = _get_neox_rotated_x(
                    k, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
                )
            else:
                k_rotated = _get_gptj_rotated_x(
                    k, qk_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
                )

        k = k * cos + k_rotated * sin

        kv_out_offs = (
            t_offs[:, None] * stride_kv_t
            + d_offs[None, :] * stride_kv_d
            + hkv * stride_kv_h
        )
        tl.store(k_ptr + kv_out_offs, k.to(k_ptr.dtype.element_ty), mask=x_mask)

        slots = tl.load(slot_mapping_ptr + t_offs, mask=t_mask)
        valid_slot = (slots >= 0) & (slots < total_num_kv_cache_tokens)
        safe_slots = tl.where(valid_slot, slots, 0)
        b_idx = safe_slots % BLOCK_SIZE
        t_slot_idx = safe_slots // BLOCK_SIZE
        cache_mask = x_mask & valid_slot[:, None]

        k = k * (1 / k_scale)
        k_cache_offs = (
            t_slot_idx[:, None] * key_cache_stride_t
            + hkv * key_cache_stride_h
            + d_offs[None, :] * key_cache_stride_d
            + b_idx[:, None] * key_cache_stride_b
        )
        tl.store(
            key_cache_ptr + k_cache_offs,
            k.to(key_cache_ptr.dtype.element_ty),
            mask=cache_mask,
        )

    else:
        # ================= V segment (no rotation, no norm) =================
        hkv = by - K_END

        if HAVE_V_SCALE:
            v_scale = tl.load(v_scale_ptr)
        else:
            v_scale = 1

        KV_HEAD_OFFS = hkv * BLOCK_D
        v_in_offs = (
            t_offs[:, None] * stride_qkv_t
            + ((Q_SIZE + KV_SIZE + KV_HEAD_OFFS) + d_offs)[None, :] * stride_qkv_d
        )
        v = tl.load(qkv_ptr + v_in_offs, mask=x_mask)

        kv_out_offs = (
            t_offs[:, None] * stride_kv_t
            + d_offs[None, :] * stride_kv_d
            + hkv * stride_kv_h
        )
        tl.store(v_ptr + kv_out_offs, v.to(v_ptr.dtype.element_ty), mask=x_mask)

        slots = tl.load(slot_mapping_ptr + t_offs, mask=t_mask)
        valid_slot = (slots >= 0) & (slots < total_num_kv_cache_tokens)
        safe_slots = tl.where(valid_slot, slots, 0)
        b_idx = safe_slots % BLOCK_SIZE
        t_slot_idx = safe_slots // BLOCK_SIZE
        cache_mask = x_mask & valid_slot[:, None]

        v = v * (1 / v_scale)
        v_cache_offs = (
            t_slot_idx[:, None] * value_cache_stride_t
            + hkv * value_cache_stride_h
            + d_offs[None, :] * value_cache_stride_d
            + b_idx[:, None] * value_cache_stride_b
        )
        tl.store(
            value_cache_ptr + v_cache_offs,
            v.to(value_cache_ptr.dtype.element_ty),
            mask=cache_mask,
        )
