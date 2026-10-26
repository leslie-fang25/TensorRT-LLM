# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perfect-router capacity proofs and strict backend eligibility checks."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.balanced_workspace import (
    BALANCED_WORKSPACE_ENV,
    balanced_assignment_capacity,
    balanced_workspace_enabled,
    validate_balanced_workspace,
)
from tensorrt_llm._torch.moe.fused_moe.communication import NVLinkOneSided
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.moe.fused_moe.routing import (
    RenormalizeMoeRoutingMethod,
    create_load_balanced_logits,
)
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("rows", [2, 6, 64, 128])
@pytest.mark.parametrize("top_k", [2, 4])
def test_capacity_matches_actual_perfect_router(rows: int, top_k: int) -> None:
    counts = torch.zeros(4, dtype=torch.int64)
    routing = RenormalizeMoeRoutingMethod(top_k=top_k)
    for sender in range(4):
        logits = create_load_balanced_logits(
            rows, 8, top_k, 4, sender, torch.device("cpu"), torch.float32, routing
        )
        selected = logits.topk(top_k, dim=-1).indices
        counts += torch.bincount((selected // 2).flatten(), minlength=4)
    capacity = balanced_assignment_capacity([rows] * 4, top_k, 4)
    assert counts.tolist() == [capacity] * 4


@pytest.mark.parametrize(
    "sizes,top_k,ep",
    [
        ([0] * 4, 2, 4),
        ([1] * 4, 2, 4),
        ([1, 1, 2, 2], 2, 4),
        ([4] * 3, 2, 4),
        ([4] * 8, 2, 8),
        ([4] * 4, 0, 4),
    ],
)
def test_unsupported_capacity_is_rejected(sizes: list[int], top_k: int, ep: int) -> None:
    with pytest.raises(NotImplementedError, match=BALANCED_WORKSPACE_ENV):
        balanced_assignment_capacity(sizes, top_k, ep)


def test_repro_capacity() -> None:
    assert balanced_assignment_capacity([16384] * 4, 2, 4) == 32768


def test_opt_in_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(BALANCED_WORKSPACE_ENV, raising=False)
    assert not balanced_workspace_enabled()
    monkeypatch.setenv(BALANCED_WORKSPACE_ENV, "1")
    assert balanced_workspace_enabled()
    monkeypatch.setenv(BALANCED_WORKSPACE_ENV, "0")
    assert not balanced_workspace_enabled()
    monkeypatch.setenv(BALANCED_WORKSPACE_ENV, "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        balanced_workspace_enabled()


@pytest.fixture
def supported_moe(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr("tensorrt_llm._utils.get_sm_version", lambda: 103)
    monkeypatch.setattr(
        "tensorrt_llm._torch.autotuner.AutoTuner.get",
        lambda: SimpleNamespace(is_tuning_mode=False),
    )
    monkeypatch.setattr(CutlassFusedMoE, "supports_alltoall", property(lambda self: True))
    backend = CutlassFusedMoE.__new__(CutlassFusedMoE)
    torch.nn.Module.__init__(backend)
    backend._weights_created = True
    backend.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    backend.dtype = torch.bfloat16
    backend.tp_size = 1
    backend.activation_type = ActivationType.Swiglu
    backend.bias = False
    backend.apply_router_weight_on_input = False
    backend.act_alpha = backend.act_beta = backend.act_clamp = None
    return SimpleNamespace(
        _enable_perfect_router=True,
        backend=backend,
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        comm=NVLinkOneSided.__new__(NVLinkOneSided),
        ep_size=4,
        mapping=SimpleNamespace(dp_size=4),
        use_dp=True,
        _using_load_balancer=lambda: False,
        enable_dwdp=False,
        model_config=SimpleNamespace(lora_config=None),
        unfinalized_combine_fn=None,
    )


def test_supported_backend(supported_moe: SimpleNamespace) -> None:
    validate_balanced_workspace(supported_moe)


def test_autotuning_is_rejected(
    supported_moe: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.autotuner.AutoTuner.get",
        lambda: SimpleNamespace(is_tuning_mode=True),
    )
    with pytest.raises(NotImplementedError, match="autotuning disabled"):
        validate_balanced_workspace(supported_moe)


@pytest.mark.parametrize(
    "field,value",
    [
        ("_enable_perfect_router", False),
        ("backend", object()),
        ("comm", object()),
        ("routing_method", object()),
        ("ep_size", 8),
        ("use_dp", False),
        ("enable_dwdp", True),
        ("unfinalized_combine_fn", object()),
    ],
)
def test_unsupported_layer_is_rejected(
    supported_moe: SimpleNamespace, field: str, value: object
) -> None:
    setattr(supported_moe, field, value)
    with pytest.raises(NotImplementedError, match=BALANCED_WORKSPACE_ENV):
        validate_balanced_workspace(supported_moe)


@pytest.mark.parametrize(
    "field,value",
    [
        ("_weights_created", False),
        ("dtype", torch.float16),
        ("tp_size", 2),
        ("bias", True),
        ("apply_router_weight_on_input", True),
        ("activation_type", ActivationType.Geglu),
        ("act_alpha", torch.ones(1)),
        ("quant_config", QuantConfig(quant_algo=QuantAlgo.NVFP4)),
    ],
)
def test_unsupported_kernel_path_is_rejected(
    supported_moe: SimpleNamespace, field: str, value: object
) -> None:
    setattr(supported_moe.backend, field, value)
    with pytest.raises(NotImplementedError, match=BALANCED_WORKSPACE_ENV):
        validate_balanced_workspace(supported_moe)
