from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.amp import autocast

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


TP_ROOT = Path(__file__).resolve().parents[1]
if str(TP_ROOT) not in sys.path:
    sys.path.insert(0, str(TP_ROOT))


from vehicle_traj.modules.metrics import ade_fde, best_mode_by_fde, minade_minfde  # noqa: E402
from vehicle_traj.modules.training import prepare_transformer_batch  # noqa: E402


def _select_prediction_mask(valid_future_mask, ego_idx, prediction_target: str):
    if prediction_target == "all_agents":
        return valid_future_mask
    if prediction_target != "ego_only":
        raise ValueError(f"Unknown prediction_target={prediction_target!r}")
    B, F_fut, N = valid_future_mask.shape
    ego = ego_idx.clamp(min=0, max=N - 1)
    ego_mask_bn = torch.zeros(B, N, dtype=torch.bool, device=valid_future_mask.device)
    ego_mask_bn[torch.arange(B, device=valid_future_mask.device), ego] = True
    ego_mask = ego_mask_bn[:, None, :].expand(B, F_fut, N)
    return valid_future_mask & ego_mask


def _unnormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # mean/std are shaped [1,1,1,2] for broadcasting
    return x * std + mean


def _remap_state_dict_prefix(state: dict[str, torch.Tensor], old: str, new: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(old):
            out[new + k[len(old) :]] = v
        else:
            out[k] = v
    return out


def _load_state_dict_compat(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """
    Backward/forward compatibility for renamed projection layers.

    Known rename(s):
      - _map_proj.*  <-> _map_proj_raw.*
      - _tl_proj.*   <-> _tl_proj_raw.*
    """
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    remapped = dict(state)

    def _maybe_remap(old_prefix: str, new_prefix: str) -> None:
        nonlocal remapped
        has_old = any(k.startswith(old_prefix) for k in state_keys)
        needs_new = any(k.startswith(new_prefix) for k in model_keys)
        missing_new = needs_new and not any(k.startswith(new_prefix) for k in state_keys)
        if has_old and missing_new:
            remapped = _remap_state_dict_prefix(remapped, old_prefix, new_prefix)

    # Checkpoint old -> model new
    _maybe_remap("_map_proj.", "_map_proj_raw.")
    _maybe_remap("_tl_proj.", "_tl_proj_raw.")
    # Checkpoint new -> model old
    _maybe_remap("_map_proj_raw.", "_map_proj.")
    _maybe_remap("_tl_proj_raw.", "_tl_proj.")

    model.load_state_dict(remapped, strict=True)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    try:
        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        run_dir = Path(os.getcwd())

    data_module = instantiate(cfg.data)
    loader = data_module.val_dataloader()

    model = instantiate(cfg.model).to(device)
    ckpt = torch.load(cfg.eval.checkpoint, map_location="cpu")
    _load_state_dict_compat(model, ckpt["model_state"])
    model.eval()

    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)

    amp = bool(getattr(cfg.eval, "amp", False)) and device.type == "cuda"

    totals = {"ade": 0.0, "fde": 0.0, "count": 0}
    totals_mm = {"minade": 0.0, "minfde": 0.0}
    with torch.inference_mode():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            min_obs_frames = int(getattr(cfg.task, "min_obs_frames", 0))
            min_fut_frames = int(getattr(cfg.task, "min_fut_frames", 0))
            inputs, target_deltas, valid_future_mask, filtered_valid_mask = prepare_transformer_batch(
                batch,
                mean=mean,
                std=std,
                obs_len=int(cfg.data.obs_len),
                future_len=int(cfg.data.future_len),
                device=device,
                placeholder_strategy=str(cfg.task.placeholder_strategy),
                ego_idx=batch.get("ego_idx"),
                min_obs_frames=min_obs_frames,
                min_fut_frames=min_fut_frames,
            )
            batch["valid_mask"] = filtered_valid_mask

            map_tokens = batch.get("map_tokens")
            tl_tokens = batch.get("tl_tokens")

            use_polyline_encoder = bool(getattr(cfg.model, "map_use_polyline_encoder", False))
            roadgraph_static = batch.get("roadgraph_static") if use_polyline_encoder else None
            padding_mask_static = batch.get("padding_mask_static") if use_polyline_encoder else None

            m = mean.view(1, 1, 2)
            s = std.view(1, 1, 2)

            if use_polyline_encoder:
                if roadgraph_static is None or padding_mask_static is None:
                    raise ValueError(
                        "cfg.model.map_use_polyline_encoder=true but batch is missing roadgraph_static/padding_mask_static"
                    )
                roadgraph_static = roadgraph_static.to(dtype=torch.float32)
                roadgraph_static = roadgraph_static.clone()
                roadgraph_static[..., 0:2] = (roadgraph_static[..., 0:2] - m.view(1, 1, 1, 2)) / s.view(1, 1, 1, 2)
                padding_mask_static = padding_mask_static.to(dtype=torch.bool)
                map_tokens = None
            elif map_tokens is not None:
                map_tokens = map_tokens.to(dtype=torch.float32)
                map_tokens = map_tokens.clone()
                map_tokens[..., 0:2] = (map_tokens[..., 0:2] - m) / s

            if tl_tokens is not None:
                tl_tokens = tl_tokens.to(dtype=torch.float32)
                tl_tokens = tl_tokens.clone()
                tl_tokens[..., 0:2] = (tl_tokens[..., 0:2] - m) / s

            with autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                out = model(
                positions=inputs,
                valid_mask=batch["valid_mask"],
                velocity=batch.get("velocity"),
                heading=batch.get("heading"),
                agent_type=batch.get("agent_type"),
                agent_map_feat=batch.get("agent_map_feat"),
                map_tokens=map_tokens,
                map_key_padding_mask=batch.get("map_key_padding_mask"),
                roadgraph_static=roadgraph_static,
                padding_mask_static=padding_mask_static,
                tl_tokens=tl_tokens,
                    tl_key_padding_mask=batch.get("tl_key_padding_mask"),
                )
                pred_deltas = out.deltas

            obs_len = int(cfg.data.obs_len)
            fut_len = int(cfg.data.future_len)
            tgt_future = inputs[:, obs_len : obs_len + fut_len] + target_deltas  # normalized
            tgt_future_m = _unnormalize(tgt_future, mean, std).float()

            mask = _select_prediction_mask(
                valid_future_mask,
                ego_idx=batch["ego_idx"],
                prediction_target=str(cfg.task.prediction_target),
            )

            if pred_deltas.dim() == 4:
                pred_future = inputs[:, obs_len : obs_len + fut_len] + pred_deltas[:, obs_len : obs_len + fut_len]
                pred_future_m = _unnormalize(pred_future, mean, std).float()
                ade, fde = ade_fde(pred_future_m, tgt_future_m, mask=mask)
                minade, minfde = ade, fde
            else:
                pred_future_modes = (
                    inputs[:, obs_len : obs_len + fut_len].unsqueeze(1) + pred_deltas[:, :, obs_len : obs_len + fut_len]
                )  # [B,M,Tf,N,2] normalized
                pred_future_modes_m = _unnormalize(pred_future_modes, mean, std).float()

                minade, minfde = minade_minfde(pred_future_modes_m, tgt_future_m, mask=mask)
                best = best_mode_by_fde(pred_future_modes_m, tgt_future_m, mask=mask)
                B, M, Tf, N, _ = pred_future_modes.shape
                idx = best[:, None, :, None, None].expand(B, 1, N, Tf, 2)
                pred_best = (
                    pred_future_modes_m.permute(0, 1, 3, 2, 4)
                    .gather(1, idx)
                    .squeeze(1)
                    .permute(0, 2, 1, 3)
                )  # [B,Tf,N,2] meters
                ade, fde = ade_fde(pred_best, tgt_future_m, mask=mask)

            bs = int(tgt_future.shape[0])
            totals["ade"] += float(ade.item()) * bs
            totals["fde"] += float(fde.item()) * bs
            totals_mm["minade"] += float(minade.item()) * bs
            totals_mm["minfde"] += float(minfde.item()) * bs
            totals["count"] += bs

    denom = max(totals["count"], 1)
    print(f"Eval (meters) ADE={totals['ade']/denom:.4f} FDE={totals['fde']/denom:.4f}")
    print(f"Eval (meters) minADE={totals_mm['minade']/denom:.4f} minFDE={totals_mm['minfde']/denom:.4f}")
    print(f"Checkpoint: {cfg.eval.checkpoint}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
