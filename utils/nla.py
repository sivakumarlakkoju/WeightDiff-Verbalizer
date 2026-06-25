"""Shared NLA utilities used across all experiment scripts."""

from __future__ import annotations

import re

import torch

NLA_AV_ID = "kitft/nla-qwen2.5-7b-L20-av"
LAYER = 20
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)

WRITE_MODULES = ["mlp.down_proj", "self_attn.o_proj"]


def normalize(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm = v.float().norm().clamp_min(1e-12)
    return (v.float() * (target_scale / norm)).to(v.dtype)


def residual_facing_svd(f, layer: int, module: str, scale: float):
    """Return (vecs [d_model, rank], S [rank]) of residual-stream-facing singular vectors for one module."""
    prefix = f"base_model.model.model.layers.{layer}.{module}"
    lora_A = f.get_tensor(f"{prefix}.lora_A.weight").float()
    lora_B = f.get_tensor(f"{prefix}.lora_B.weight").float()

    if module in WRITE_MODULES:
        # δW = scale · B @ A ; left singular vectors U ∈ R^{d_model}
        Q, R = torch.linalg.qr(lora_B)
        U_small, S, _ = torch.linalg.svd(scale * R @ lora_A, full_matrices=False)
        vecs = Q @ U_small
    else:
        # δW = scale · B @ A ; right singular vectors V ∈ R^{d_model}
        Q_a, R_a = torch.linalg.qr(lora_A.T)
        _, S, Vh_small = torch.linalg.svd(scale * lora_B @ R_a.T, full_matrices=False)
        vecs = Q_a @ Vh_small.T
    return vecs, S
