# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in capacity contract for the validated perfect-router CUTLASS path."""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configurable_moe import ConfigurableMoE

BALANCED_WORKSPACE_ENV = "TRTLLM_MOE_ENABLE_BALANCED_WORKSPACE"


def balanced_workspace_enabled() -> bool:
    """Read the opt-in switch before constructing a MoE layer."""
    value = os.environ.get(BALANCED_WORKSPACE_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{BALANCED_WORKSPACE_ENV} must be 0 or 1")
    return value == "1"


def validate_balanced_workspace(moe: "ConfigurableMoE") -> None:
    """Reject configurations outside the supported capacity/layout contract."""
    import torch

    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.utils import ActivationType
    from tensorrt_llm._utils import get_sm_version

    from .communication import NVLinkOneSided
    from .fused_moe_cutlass import CutlassFusedMoE
    from .routing import RenormalizeMoeRoutingMethod

    conditions = (
        (moe._enable_perfect_router, "ENABLE_PERFECT_ROUTER=1"),
        (type(moe.backend) is CutlassFusedMoE, "the CUTLASS implementation"),
        (get_sm_version() == 103, "GB300 (SM103)"),
        (type(moe.routing_method) is RenormalizeMoeRoutingMethod, "flat Renormalize routing"),
        (type(moe.comm) is NVLinkOneSided, "NVLinkOneSided communication"),
        (moe.ep_size == 4 and moe.mapping.dp_size == 4 and moe.use_dp, "attention-DP4 / EP4"),
        (not moe._using_load_balancer(), "EPLB disabled"),
        (not moe.enable_dwdp, "DWDP disabled"),
        (moe.model_config.lora_config is None, "LoRA disabled"),
        (moe.unfinalized_combine_fn is None, "ordinary finalized output"),
        (not AutoTuner.get().is_tuning_mode, "autotuning disabled during balanced execution"),
    )
    for supported, requirement in conditions:
        if not supported:
            raise NotImplementedError(f"{BALANCED_WORKSPACE_ENV}=1 requires {requirement}")

    backend = moe.backend
    if not backend._weights_created:
        raise NotImplementedError(
            f"{BALANCED_WORKSPACE_ENV}=1 requires a resolved, initialized quantization path"
        )
    conditions = (
        (backend.supports_alltoall, "the open-source CUTLASS build"),
        (backend.has_fp8_qdq, "per-tensor FP8 quantization"),
        (backend.dtype == torch.bfloat16, "BF16 model activations/output"),
        (backend.tp_size == 1, "MoE TP1"),
        (backend.activation_type == ActivationType.Swiglu, "ordinary SwiGLU"),
        (not backend.bias, "expert bias disabled"),
        (not backend.apply_router_weight_on_input, "router weights applied at finalize"),
        (
            backend.act_alpha is None and backend.act_beta is None and backend.act_clamp is None,
            "default activation parameters",
        ),
    )
    for supported, requirement in conditions:
        if not supported:
            raise NotImplementedError(f"{BALANCED_WORKSPACE_ENV}=1 requires {requirement}")


def balanced_assignment_capacity(all_rank_num_tokens: list[int], top_k: int, ep_size: int) -> int:
    """Bound local rows for equal, nonempty, divisible flat-router batches.

    Each sender cycles its assignments across EP ranks. Divisibility makes
    every sender contribute exactly n * top_k / ep_size to every receiver.
    Padding, unequal batches and partial cycles are intentionally unsupported.
    """
    if ep_size != 4 or len(all_rank_num_tokens) != ep_size or top_k <= 0:
        raise NotImplementedError(
            f"{BALANCED_WORKSPACE_ENV}=1 requires four sender ranks and positive top-k"
        )
    n = all_rank_num_tokens[0]
    if n <= 0 or any(count != n for count in all_rank_num_tokens) or n * top_k % ep_size:
        raise NotImplementedError(
            f"{BALANCED_WORKSPACE_ENV}=1 requires equal positive token counts and "
            "per-sender token_count * top_k divisible by EP"
        )
    return sum(all_rank_num_tokens) * top_k // ep_size
