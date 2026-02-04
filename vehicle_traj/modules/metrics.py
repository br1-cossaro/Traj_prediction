from __future__ import annotations

from typing import Optional, Tuple

import torch


def masked_mse(
    pred: torch.Tensor,  # [B, F, N, 2]
    target: torch.Tensor,  # [B, F, N, 2]
    mask: Optional[torch.Tensor] = None,  # [B, F, N]
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    diff2 = (pred - target) ** 2
    per = diff2.sum(dim=-1)  # [B, F, N]
    if mask is None:
        return per.mean()
    m = mask.to(dtype=per.dtype)
    denom = m.sum().clamp(min=eps)
    return (per * m).sum() / denom


def masked_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    diff = pred - target
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + eps)  # [...]
    if mask is None:
        return dist.mean()
    m = mask.to(dtype=dist.dtype)
    denom = m.sum().clamp(min=eps)
    return (dist * m).sum() / denom


def ade_fde(
    pred: torch.Tensor,  # [B, F, N, 2]
    target: torch.Tensor,  # [B, F, N, 2]
    mask: Optional[torch.Tensor] = None,  # [B, F, N]
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    ade = masked_l2(pred, target, mask=mask, eps=eps)
    fde_mask = mask[:, -1] if mask is not None else None
    fde = masked_l2(pred[:, -1], target[:, -1], mask=fde_mask, eps=eps)
    return ade, fde


def best_mode_by_fde(
    pred_modes: torch.Tensor,  # [B, M, F, N, 2]
    target: torch.Tensor,  # [B, F, N, 2]
    mask: Optional[torch.Tensor] = None,  # [B, F, N]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns best mode index per (B,N) by minimizing final displacement error.
    """
    if pred_modes.dim() != 5:
        raise ValueError(f"pred_modes must be [B,M,F,N,2], got {tuple(pred_modes.shape)}")
    if target.dim() != 4:
        raise ValueError(f"target must be [B,F,N,2], got {tuple(target.shape)}")
    if pred_modes.shape[0] != target.shape[0] or pred_modes.shape[2:] != target.shape[1:]:
        raise ValueError(f"shape mismatch: pred_modes {pred_modes.shape} vs target {target.shape}")

    diff = pred_modes[:, :, -1] - target[:, None, -1]  # [B,M,N,2]
    fde = torch.sqrt(torch.sum(diff * diff, dim=-1) + eps)  # [B,M,N]

    if mask is not None:
        fde_mask = mask[:, -1].to(dtype=torch.bool)  # [B,N]
        fde = fde.masked_fill(~fde_mask[:, None, :], float("inf"))

    return fde.argmin(dim=1)  # [B,N]


def minade_minfde(
    pred_modes: torch.Tensor,  # [B, M, F, N, 2]
    target: torch.Tensor,  # [B, F, N, 2]
    mask: Optional[torch.Tensor] = None,  # [B, F, N]
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    minADE/minFDE over modes, with best-mode selection per agent by FDE.
    """
    best = best_mode_by_fde(pred_modes, target, mask=mask, eps=eps)  # [B,N]
    B, M, F, N, _ = pred_modes.shape

    idx = best[:, None, None, :, None].expand(B, 1, F, N, 2)
    pred_best = pred_modes.gather(1, idx).squeeze(1)  # [B,F,N,2]
    return ade_fde(pred_best, target, mask=mask, eps=eps)


__all__ = ["masked_mse", "ade_fde", "best_mode_by_fde", "minade_minfde"]
