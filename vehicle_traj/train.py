from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from math import inf

import torch
import torch.nn.functional as F

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


# we have to ensure `trajectory_prediction/` is on sys.path so we can import sibling modules
TP_ROOT = Path(__file__).resolve().parents[1]
if str(TP_ROOT) not in sys.path:
    sys.path.insert(0, str(TP_ROOT))


from vehicle_traj.modules.metrics import ade_fde, best_mode_by_fde, masked_mse, minade_minfde  # noqa: E402
from vehicle_traj.modules.training import (  # noqa: E402
    compute_norm_stats,
    prepare_transformer_batch,
)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_run_metadata(run_dir: Path, cfg: DictConfig) -> None:
    """
    Make runs self-contained and comparable, even when using `train.run_dir=...`
    (which bypasses Hydra's default output directory that normally creates `.hydra/`).
    """
    hydra_dir = _ensure_dir(run_dir / ".hydra")

    # Match Hydra convention: config.yaml + overrides.yaml (+ hydra.yaml when available).
    try:
        OmegaConf.save(config=cfg, f=str(hydra_dir / "config.yaml"))
    except Exception:
        pass

    try:
        hc = HydraConfig.get()
    except Exception:
        hc = None

    if hc is not None:
        try:
            OmegaConf.save(config=hc.cfg, f=str(hydra_dir / "hydra.yaml"))
        except Exception:
            pass

        try:
            overrides = list(getattr(hc.overrides, "task", []) or [])
            (hydra_dir / "overrides.yaml").write_text(
                "".join(f"- {o}\n" for o in overrides),
                encoding="utf-8",
            )
        except Exception:
            pass

    # Extra convenience: record argv so we can reproduce the exact invocation.
    try:
        (run_dir / "train_cmd.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    except Exception:
        pass


def _as_path(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def _load_last_seconds(metrics_path: Path) -> float:
    """
    When resuming a run, keep `seconds` monotonically increasing by continuing from
    the last recorded value in metrics.csv.
    """
    if not metrics_path.exists():
        return 0.0
    try:
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return 0.0
        return float(rows[-1].get("seconds", 0.0) or 0.0)
    except Exception:
        return 0.0


def _load_best_from_metrics(
    metrics_path: Path,
    *,
    metric: str,
    mode: str,
) -> tuple[int, float]:
    """
    Reconstruct best (epoch, value) from an existing metrics.csv.
    Returns (0, +/-inf) if no data.
    """
    if not metrics_path.exists():
        return (0, inf if mode == "min" else -inf)
    try:
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        best_epoch = 0
        best_value = inf if mode == "min" else -inf
        for r in rows:
            if metric not in r or r[metric] in ("", None):
                continue
            v = float(r[metric])
            e = int(float(r.get("epoch", 0) or 0))
            if mode == "min":
                if v < best_value:
                    best_value, best_epoch = v, e
            elif mode == "max":
                if v > best_value:
                    best_value, best_epoch = v, e
            else:
                raise ValueError(f"Unknown select_mode={mode!r} (expected 'min' or 'max')")
        return best_epoch, best_value
    except Exception:
        return (0, inf if mode == "min" else -inf)


def _move_optimizer_state_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _remap_state_dict_prefix(
    state: dict[str, torch.Tensor],
    old: str,
    new: str,
) -> dict[str, torch.Tensor]:
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


def _select_prediction_mask(
    valid_future_mask: torch.Tensor,  # [B, F_fut, N]
    ego_idx: torch.Tensor,  # [B]
    prediction_target: str,
) -> torch.Tensor:
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


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    torch.manual_seed(int(cfg.seed))

    run_dir_cfg = str(getattr(cfg.train, "run_dir", "")).strip()
    resume_from_cfg = str(getattr(cfg.train, "resume_from", "")).strip()

    if run_dir_cfg:
        run_dir = _as_path(run_dir_cfg)
    elif resume_from_cfg:
        # Convenience: if resume_from points to .../<run>/checkpoints/epoch_XXX.pt
        # and no run_dir is provided, infer run_dir from the checkpoint path.
        rp = _as_path(resume_from_cfg)
        run_dir = rp.parent.parent if rp.name.endswith(".pt") and rp.parent.name == "checkpoints" else rp.parent
    else:
        try:
            run_dir = Path(HydraConfig.get().runtime.output_dir)
        except Exception:
            run_dir = Path(os.getcwd())

    _ensure_dir(run_dir / "checkpoints")
    _ensure_dir(run_dir / "logs")
    _write_run_metadata(run_dir, cfg)

    print(f"Run dir: {run_dir}")
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    data_module = instantiate(cfg.data)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    model = instantiate(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    metrics_path = run_dir / "metrics.csv"
    seconds_offset = _load_last_seconds(metrics_path)
    start_time = time.perf_counter() - seconds_offset

    start_epoch = 1
    ckpt = None
    if resume_from_cfg:
        resume_path = _as_path(resume_from_cfg)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_from does not exist: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")

        if "model_state" in ckpt:
            _load_state_dict_compat(model, ckpt["model_state"])
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                _move_optimizer_state_to(optimizer, device)
            except Exception:
                pass
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1

        # Ensure optimizer settings match the current config (it can be useful when resuming with overrides).
        for pg in optimizer.param_groups:
            pg["lr"] = float(cfg.train.lr)
            pg["weight_decay"] = float(cfg.train.weight_decay)

    if ckpt is not None and "mean" in ckpt and "std" in ckpt:
        mean = torch.as_tensor(ckpt["mean"], dtype=torch.float32, device=device)
        std = torch.as_tensor(ckpt["std"], dtype=torch.float32, device=device)
    else:
        mean, std = compute_norm_stats(
            train_loader,
            obs_len=int(cfg.data.obs_len),
            device=device,
        )

    def run_epoch(loader, *, is_train: bool):
        model.train(is_train)
        total = {
            "loss": 0.0,
            "reg_loss": 0.0,
            "mode_ce": 0.0,
            "ade": 0.0,
            "fde": 0.0,
            "minade": 0.0,
            "minfde": 0.0,
            "count": 0,
        }

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
            mode_logits = out.mode_logits

            obs_len = int(cfg.data.obs_len)
            fut_len = int(cfg.data.future_len)
            tgt_future = inputs[:, obs_len : obs_len + fut_len] + target_deltas  # [B,Tf,N,2]
            tgt_future_m = tgt_future * s.view(1, 1, 1, 2) + m.view(1, 1, 1, 2)

            mask = _select_prediction_mask(
                valid_future_mask,
                ego_idx=batch["ego_idx"],
                prediction_target=str(cfg.task.prediction_target),
            )

            w_mode_ce = float(getattr(cfg.train, "w_mode_ce", 0.0))
            mode_ce = torch.tensor(0.0, device=device)

            if pred_deltas.dim() == 4:
                pred_future = inputs[:, obs_len : obs_len + fut_len] + pred_deltas[:, obs_len : obs_len + fut_len]
                reg_loss = masked_mse(pred_future, tgt_future, mask=mask)
                loss = reg_loss
                pred_future_m = pred_future * s.view(1, 1, 1, 2) + m.view(1, 1, 1, 2)
                ade, fde = ade_fde(pred_future_m, tgt_future_m, mask=mask)
                minade, minfde = ade, fde
            elif pred_deltas.dim() == 5:
                # [B,M,F,N,2] -> [B,M,Tf,N,2]
                pred_future_modes = (
                    inputs[:, obs_len : obs_len + fut_len].unsqueeze(1)
                    + pred_deltas[:, :, obs_len : obs_len + fut_len]
                )
                pred_future_modes_m = pred_future_modes * s.view(1, 1, 1, 1, 2) + m.view(1, 1, 1, 1, 2)
                minade, minfde = minade_minfde(pred_future_modes_m, tgt_future_m, mask=mask)

                best = best_mode_by_fde(pred_future_modes_m, tgt_future_m, mask=mask)  # [B,N]
                B, M, Tf, N, _ = pred_future_modes.shape
                idx = best[:, None, :, None, None].expand(B, 1, N, Tf, 2)
                pred_best = pred_future_modes.permute(0, 1, 3, 2, 4).gather(1, idx).squeeze(1)  # [B,N,Tf,2]
                pred_best = pred_best.permute(0, 2, 1, 3).contiguous()  # [B,Tf,N,2]
                pred_best_m = pred_best * s.view(1, 1, 1, 2) + m.view(1, 1, 1, 2)

                reg_loss = masked_mse(pred_best, tgt_future, mask=mask)

                if mode_logits is None:
                    raise RuntimeError("Multimodal output requires mode_logits")
                # Cross-entropy per agent (only for agents that have any future GT)
                agent_has_gt = mask.any(dim=1)  # [B,N]
                ce_per = F.cross_entropy(
                    mode_logits.reshape(B * N, M),
                    best.reshape(B * N),
                    reduction="none",
                ).view(B, N)
                denom = agent_has_gt.to(dtype=ce_per.dtype).sum().clamp(min=1.0)
                mode_ce = (ce_per * agent_has_gt.to(dtype=ce_per.dtype)).sum() / denom

                loss = reg_loss + w_mode_ce * mode_ce
                ade, fde = ade_fde(pred_best_m, tgt_future_m, mask=mask)
            else:
                raise ValueError(f"Unexpected pred_deltas shape: {tuple(pred_deltas.shape)}")

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg.train.max_grad_norm),
                )
                optimizer.step()

            bs = int(tgt_future.shape[0])
            total["loss"] += float(loss.item()) * bs
            total["reg_loss"] += float(reg_loss.item()) * bs
            total["mode_ce"] += float(mode_ce.item()) * bs
            total["ade"] += float(ade.item()) * bs
            total["fde"] += float(fde.item()) * bs
            total["minade"] += float(minade.item()) * bs
            total["minfde"] += float(minfde.item()) * bs
            total["count"] += bs

        denom = max(total["count"], 1)
        return {k: v / denom for k, v in total.items() if k != "count"}

    fieldnames = [
        "epoch",
        "seconds",
        "lr",
        "train_loss",
        "train_reg_loss",
        "train_mode_ce",
        "train_ade",
        "train_fde",
        "train_minade",
        "train_minfde",
        "val_loss",
        "val_reg_loss",
        "val_mode_ce",
        "val_ade",
        "val_fde",
        "val_minade",
        "val_minfde",
    ]

    epochs = int(cfg.train.epochs)
    if start_epoch > epochs:
        print(f"Nothing to do: start_epoch={start_epoch} > train.epochs={epochs} (already finished?).")
        return

    select_metric = str(getattr(cfg.train, "select_metric", "val_minfde")).strip() or "val_minfde"
    select_mode = str(getattr(cfg.train, "select_mode", "min")).strip() or "min"
    early_stop_patience = int(getattr(cfg.train, "early_stop_patience", 0) or 0)
    early_stop_min_delta = float(getattr(cfg.train, "early_stop_min_delta", 0.0) or 0.0)
    save_best = bool(getattr(cfg.train, "save_best", True))

    append = bool(resume_from_cfg) and metrics_path.exists() and metrics_path.stat().st_size > 0
    mode = "a" if append else "w"

    best_epoch, best_value = _load_best_from_metrics(metrics_path, metric=select_metric, mode=select_mode)
    if append and best_epoch > 0:
        print(f"Resuming best-so-far: {select_metric}={best_value:.6f} at epoch {best_epoch:03d}")
    no_improve = 0

    with metrics_path.open(mode, newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, epochs + 1):
            train_metrics = run_epoch(train_loader, is_train=True)
            val_metrics = run_epoch(val_loader, is_train=False)

            lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "seconds": round(time.perf_counter() - start_time, 3),
                "lr": lr,
                "train_loss": float(train_metrics["loss"]),
                "train_reg_loss": float(train_metrics["reg_loss"]),
                "train_mode_ce": float(train_metrics["mode_ce"]),
                "train_ade": float(train_metrics["ade"]),
                "train_fde": float(train_metrics["fde"]),
                "train_minade": float(train_metrics["minade"]),
                "train_minfde": float(train_metrics["minfde"]),
                "val_loss": float(val_metrics["loss"]),
                "val_reg_loss": float(val_metrics["reg_loss"]),
                "val_mode_ce": float(val_metrics["mode_ce"]),
                "val_ade": float(val_metrics["ade"]),
                "val_fde": float(val_metrics["fde"]),
                "val_minade": float(val_metrics["minade"]),
                "val_minfde": float(val_metrics["minfde"]),
            }
            writer.writerow(row)
            f_csv.flush()
            try:
                os.fsync(f_csv.fileno())
            except OSError:
                pass

            # Checkpoint selection logic.
            cur_value = float(row.get(select_metric))
            improved = False
            if select_mode == "min":
                improved = cur_value < (best_value - early_stop_min_delta)
            elif select_mode == "max":
                improved = cur_value > (best_value + early_stop_min_delta)
            else:
                raise ValueError(f"Unknown select_mode={select_mode!r} (expected 'min' or 'max')")

            if improved:
                best_value = cur_value
                best_epoch = int(epoch)
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.6f} ade={train_metrics['ade']:.4f} fde={train_metrics['fde']:.4f} | "
                f"val loss={val_metrics['loss']:.6f} ade={val_metrics['ade']:.4f} fde={val_metrics['fde']:.4f} | "
                f"best {select_metric}={best_value:.4f}@{best_epoch:03d}"
            )

            if epoch % int(cfg.train.ckpt_every) == 0 or epoch == int(cfg.train.epochs) or improved:
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "mean": mean.detach().cpu(),
                    "std": std.detach().cpu(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "best": {
                        "metric": select_metric,
                        "mode": select_mode,
                        "value": best_value,
                        "epoch": best_epoch,
                    },
                }
                # Always keep per-epoch checkpoints at the requested cadence.
                if epoch % int(cfg.train.ckpt_every) == 0 or epoch == int(cfg.train.epochs):
                    torch.save(ckpt, run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
                # Additionally keep a convenient "best.pt" symlink-like checkpoint.
                if save_best and improved:
                    torch.save(ckpt, run_dir / "checkpoints" / "best.pt")

            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch:03d}: no improvement in {early_stop_patience} epochs "
                    f"(best {select_metric}={best_value:.6f}@{best_epoch:03d})."
                )
                break


if __name__ == "__main__":
    main()
