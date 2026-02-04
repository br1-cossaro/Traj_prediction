from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.amp import autocast


TP_ROOT = Path(__file__).resolve().parents[1]
if str(TP_ROOT) not in sys.path:
    sys.path.insert(0, str(TP_ROOT))

from vehicle_traj.modules.metrics import ade_fde, best_mode_by_fde, minade_minfde  # noqa: E402
from vehicle_traj.modules.training import prepare_transformer_batch  # noqa: E402


def _as_path(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a vehicle_traj checkpoint using the cfg stored in the checkpoint.")
    ap.add_argument("--checkpoint", required=True, help="Path to epoch_XXX.pt or best.pt")
    ap.add_argument("--data-root", default="", help="Optional override for cfg.data.root (NPZ dir or file).")
    ap.add_argument("--split-path", default="", help="Optional override for cfg.data.split_path (fixed split npz).")
    ap.add_argument("--device", default="", help="cpu|cuda (default: use cfg.device).")
    ap.add_argument("--prediction-target", default="", choices=["", "all_agents", "ego_only"])
    ap.add_argument("--batch-size", type=int, default=0, help="Override cfg.data.batch_size if >0.")
    ap.add_argument("--num-workers", type=int, default=0, help="Override cfg.data.num_workers if >0.")
    ap.add_argument("--amp", action="store_true", help="Enable AMP if device=cuda.")
    ap.add_argument("--max-batches", type=int, default=0, help="If >0, evaluate only first N batches.")
    ap.add_argument("--json-out", default="", help="Optional JSON output path.")
    ap.add_argument("--map-max-radius-m", type=float, default=None, help="Override cfg.data.map_max_radius_m (<=0 disables).")
    ap.add_argument("--map-max-polylines", type=int, default=None, help="Override cfg.data.map_max_polylines.")
    ap.add_argument("--tl-max-radius-m", type=float, default=None, help="Override cfg.data.tl_max_radius_m (<=0 disables).")
    ap.add_argument("--tl-max-lights", type=int, default=None, help="Override cfg.data.tl_max_lights.")
    ap.add_argument("--min-obs-frames", type=int, default=-1, help="Override cfg.task.min_obs_frames if >=0.")
    ap.add_argument("--min-fut-frames", type=int, default=-1, help="Override cfg.task.min_fut_frames if >=0.")
    args = ap.parse_args()

    ckpt_path = _as_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_raw = ckpt.get("cfg")
    if cfg_raw is None:
        raise KeyError("Checkpoint missing 'cfg'. Re-train with vehicle_traj/train.py to include cfg in checkpoints.")

    cfg = OmegaConf.create(cfg_raw)

    if str(args.data_root).strip():
        cfg.data.root = str(_as_path(args.data_root))
    if str(args.split_path).strip():
        cfg.data.split_path = str(_as_path(args.split_path))
    if str(args.device).strip():
        cfg.device = str(args.device).strip()
    if str(args.prediction_target).strip():
        cfg.task.prediction_target = str(args.prediction_target).strip()
    if int(args.batch_size) > 0:
        cfg.data.batch_size = int(args.batch_size)
    if int(args.num_workers) > 0:
        cfg.data.num_workers = int(args.num_workers)
    cfg.eval.amp = bool(args.amp)

    if args.map_max_radius_m is not None:
        cfg.data.map_max_radius_m = float(args.map_max_radius_m)
    if args.map_max_polylines is not None:
        cfg.data.map_max_polylines = int(args.map_max_polylines)
    if args.tl_max_radius_m is not None:
        cfg.data.tl_max_radius_m = float(args.tl_max_radius_m)
    if args.tl_max_lights is not None:
        cfg.data.tl_max_lights = int(args.tl_max_lights)
    if int(args.min_obs_frames) >= 0:
        cfg.task.min_obs_frames = int(args.min_obs_frames)
    if int(args.min_fut_frames) >= 0:
        cfg.task.min_fut_frames = int(args.min_fut_frames)

    device = torch.device(str(cfg.device))
    data_module = instantiate(cfg.data)
    loader = data_module.val_dataloader()

    model = instantiate(cfg.model).to(device)
    _load_state_dict_compat(model, ckpt["model_state"])
    model.eval()

    mean = ckpt["mean"].to(device)
    std = ckpt["std"].to(device)
    amp = bool(cfg.eval.amp) and device.type == "cuda"

    totals = {"ade": 0.0, "fde": 0.0, "minade": 0.0, "minfde": 0.0, "count": 0}
    with torch.inference_mode():
        for batch_i, batch in enumerate(loader):
            if int(args.max_batches) > 0 and batch_i >= int(args.max_batches):
                break
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
            tgt_future = inputs[:, obs_len : obs_len + fut_len] + target_deltas
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
                pred_future_modes = inputs[:, obs_len : obs_len + fut_len].unsqueeze(1) + pred_deltas[:, :, obs_len : obs_len + fut_len]
                pred_future_modes_m = _unnormalize(pred_future_modes, mean, std).float()
                minade, minfde = minade_minfde(pred_future_modes_m, tgt_future_m, mask=mask)

                best = best_mode_by_fde(pred_future_modes_m, tgt_future_m, mask=mask)
                B, M, Tf, N, _ = pred_future_modes_m.shape
                idx = best[:, None, :, None, None].expand(B, 1, N, Tf, 2)
                pred_best = pred_future_modes_m.permute(0, 1, 3, 2, 4).gather(1, idx).squeeze(1).permute(0, 2, 1, 3)
                ade, fde = ade_fde(pred_best, tgt_future_m, mask=mask)

            bs = int(tgt_future.shape[0])
            totals["ade"] += float(ade.item()) * bs
            totals["fde"] += float(fde.item()) * bs
            totals["minade"] += float(minade.item()) * bs
            totals["minfde"] += float(minfde.item()) * bs
            totals["count"] += bs

    denom = max(int(totals["count"]), 1)
    out = {
        "checkpoint": str(ckpt_path),
        "prediction_target": str(cfg.task.prediction_target),
        "device": str(device),
        "amp": bool(amp),
        "ade_m": totals["ade"] / denom,
        "fde_m": totals["fde"] / denom,
        "minade_m": totals["minade"] / denom,
        "minfde_m": totals["minfde"] / denom,
        "count": denom,
    }

    print(f"Eval (meters) ADE={out['ade_m']:.4f} FDE={out['fde_m']:.4f}")
    print(f"Eval (meters) minADE={out['minade_m']:.4f} minFDE={out['minfde_m']:.4f}")
    print(f"Checkpoint: {out['checkpoint']}")

    if str(args.json_out).strip():
        p = _as_path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
