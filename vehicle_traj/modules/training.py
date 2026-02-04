from __future__ import annotations

from typing import Dict, Tuple

import torch


@torch.no_grad()
def compute_norm_stats(
    train_loader,
    obs_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean/std over observed frames only, using valid_mask.
    Returns mean/std shaped [1, 1, 1, 2] for broadcasting.
    """
    sum_vec = torch.zeros(2, device=device)
    sum_sq = torch.zeros(2, device=device)
    count = 0

    for batch in train_loader:
        pos = batch["positions"].to(device)[:, :obs_len]  # [B, obs_len, N, 2]
        mask = batch["valid_mask"].to(device)[:, :obs_len]  # [B, obs_len, N]
        valid = mask.unsqueeze(-1)  # [B, obs_len, N, 1]
        vals = pos[valid.expand_as(pos)].view(-1, 2)  # [K, 2]
        if vals.numel() == 0:
            continue
        sum_vec += vals.sum(dim=0)
        sum_sq += (vals ** 2).sum(dim=0)
        count += int(vals.shape[0])

    denom = max(count, 1)
    mean = sum_vec / denom
    var = sum_sq / denom - mean ** 2
    std = torch.sqrt(var.clamp(min=1e-6))
    return mean.view(1, 1, 1, 2), std.view(1, 1, 1, 2)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def prepare_transformer_batch(
    batch: Dict[str, torch.Tensor],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    obs_len: int,
    future_len: int,
    device: torch.device,
    placeholder_strategy: str = "last_obs",
    ego_idx: torch.Tensor | None = None,
    min_obs_frames: int = 0,
    min_fut_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        inputs: [B, obs_len+future_len, N, 2] normalized; future slots filled per strategy
        target_deltas: [B, future_len, N, 2] normalized deltas to predict (relative to last obs)
        valid_future_mask: [B, future_len, N] bool
        filtered_valid_mask: [B, obs_len+future_len, N] bool (after optional agent filtering)
    """
    positions = batch["positions"].to(device)  # [B, F, N, 2]
    valid_mask = batch["valid_mask"].to(device)  # [B, F, N]

    F_total = obs_len + future_len
    if positions.shape[1] < F_total:
        raise ValueError(
            f"Need at least {F_total} frames, got {positions.shape[1]}."
        )

    mean = mean.to(device)
    std = std.to(device)
    pos_norm = normalize(positions, mean, std)  # [B, F, N, 2]

    # Optional: filter agents by minimum valid obs/future frames.
    min_obs = max(int(min_obs_frames), 0)
    min_fut = max(int(min_fut_frames), 0)
    if min_obs > 0 or min_fut > 0:
        B, F, N = valid_mask.shape
        obs_valid = valid_mask[:, :obs_len].sum(dim=1)  # [B, N]
        fut_valid = valid_mask[:, obs_len:obs_len + future_len].sum(dim=1)  # [B, N]

        keep = torch.ones((B, N), dtype=torch.bool, device=valid_mask.device)
        if min_obs > 0:
            keep &= obs_valid >= min_obs
        if min_fut > 0:
            keep &= fut_valid >= min_fut

        if ego_idx is not None:
            ego = ego_idx.to(valid_mask.device).clamp(min=0, max=N - 1)
            keep[torch.arange(B, device=valid_mask.device), ego] = True

        valid_mask = valid_mask & keep[:, None, :]

    obs = pos_norm[:, :obs_len]  # [B, obs_len, N, 2]
    fut = pos_norm[:, obs_len:obs_len + future_len]  # [B, future_len, N, 2]
    valid_future_mask = valid_mask[:, obs_len:obs_len + future_len]  # [B, future_len, N]

    last_obs = obs[:, obs_len - 1:obs_len]  # [B, 1, N, 2]
    target_deltas = fut - last_obs  # [B, future_len, N, 2]

    inputs = pos_norm[:, :F_total].clone()
    if placeholder_strategy == "last_obs":
        inputs[:, obs_len:] = last_obs.expand(-1, future_len, -1, -1)
    else:
        raise ValueError(f"Unknown placeholder_strategy={placeholder_strategy!r}")

    return inputs, target_deltas, valid_future_mask, valid_mask


__all__ = ["compute_norm_stats", "prepare_transformer_batch"]
