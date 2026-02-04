from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from datasets.waymo_scene_dataset import WaymoSceneDataset, waymo_collate_fn


@dataclass
class WaymoSceneDataModule:
    """
    Minimal datamodule around WaymoSceneDataset with an internal train/val split.
    """

    root: str
    batch_size: int = 4
    num_workers: int = 0
    val_ratio: float = 0.2
    seed: int = 0
    split_path: str = ""

    max_agents: Optional[int] = None
    num_frames: Optional[int] = None
    obs_len: int = 11
    future_len: int = 80
    relative_to_ego: bool = True

    #  filtering of map / traffic-light tokens after ego-centering.
    #
    # Motivation: map polylines and traffic lights can be far away from the
    # ego and dominate the coordinate scale, making them less useful and
    # potentially harming normalization. Keeping only nearby tokens increases
    # relevance and makes map-vs-no-map comparisons more meaningful.
    #
    # Set radius <= 0 to disable radius filtering.
    map_max_radius_m: float = 0.0
    map_max_polylines: int = 256
    tl_max_radius_m: float = 0.0
    tl_max_lights: int = 64
    tl_history_frames: int = 1

    def _add_map_and_signal_tokens(self, base: dict, batch) -> dict:
        """
        Optionally stacks map/signal tensors from per-scene samples and produces compact token sets:
          - map_tokens: one token per polyline (centroid + type)
          - tl_tokens: tokens at last observed frame (x,y,state) or a short history window
        """
        B = int(base["positions"].shape[0])
        F = int(base["positions"].shape[1])

        ego_origin = base.get("ego_origin")  # [B,2]
        origin_xy = None
        if bool(self.relative_to_ego) and torch.is_tensor(ego_origin):
            origin_xy = ego_origin.to(dtype=torch.float32).view(B, 1, 1, 2)

        # --- Static roadgraph polylines ---
        if any("roadgraph_static" in item for item in batch):
            rg0 = None
            for item in batch:
                if "roadgraph_static" in item:
                    rg0 = item["roadgraph_static"]
                    break
            if rg0 is None:
                raise RuntimeError("Expected roadgraph_static in batch but none found.")
            rg0 = torch.as_tensor(rg0, dtype=torch.float32)
            if rg0.dim() != 3:
                raise ValueError(f"roadgraph_static must be [Gs,P,C], got {tuple(rg0.shape)}")
            Gs, P, C = rg0.shape

            roadgraph = torch.zeros(B, Gs, P, C, dtype=torch.float32)
            padding_mask = torch.ones(B, Gs, P, dtype=torch.bool)  # True = padding

            for b_idx, item in enumerate(batch):
                if "roadgraph_static" not in item:
                    continue
                rg = torch.as_tensor(item["roadgraph_static"], dtype=torch.float32)
                if rg.shape != (Gs, P, C):
                    raise ValueError(f"roadgraph_static shape mismatch: expected {(Gs,P,C)}, got {tuple(rg.shape)}")
                roadgraph[b_idx] = rg

                if "padding_mask_static" in item:
                    pm = torch.as_tensor(item["padding_mask_static"], dtype=torch.bool)
                    padding_mask[b_idx] = pm
                elif "roadgraph_static_mask" in item:
                    m = torch.as_tensor(item["roadgraph_static_mask"], dtype=torch.bool)
                    padding_mask[b_idx] = ~m
                else:
                    padding_mask[b_idx] = False

            if origin_xy is not None:
                valid_pts = (~padding_mask).unsqueeze(-1)  # [B,Gs,P,1]
                xy = (roadgraph[:, :, :, 0:2] - origin_xy) * valid_pts.to(dtype=roadgraph.dtype)
                roadgraph[:, :, :, 0:2] = xy

            base["roadgraph_static"] = roadgraph
            base["padding_mask_static"] = padding_mask

            # One token per polyline: mean(xy) over valid points + type_id.
            valid = (~padding_mask).to(dtype=torch.float32).unsqueeze(-1)  # [B,Gs,P,1]
            denom = valid.sum(dim=2).clamp(min=1.0)  # [B,Gs,1]
            mean_xy = (roadgraph[:, :, :, 0:2] * valid).sum(dim=2) / denom  # [B,Gs,2]
            type_id = roadgraph[:, :, 0, 2:3] if C >= 3 else torch.zeros(B, Gs, 1, dtype=torch.float32)
            map_tokens = torch.cat([mean_xy, type_id], dim=-1)  # [B,Gs,3]
            map_key_padding_mask = padding_mask.all(dim=2)  # [B,Gs]

            # Optional: keep only nearby/top-k polylines.
            max_polylines = max(1, int(getattr(self, "map_max_polylines", 256)))
            max_radius_m = float(getattr(self, "map_max_radius_m", 0.0))
            if max_radius_m > 0.0 or max_polylines < int(map_tokens.shape[1]):
                # Distances computed in ego-centric coordinates if relative_to_ego=true.
                d = torch.norm(map_tokens[:, :, 0:2], dim=-1)  # [B,Gs]
                keep_mask = ~map_key_padding_mask
                if max_radius_m > 0.0:
                    keep_mask = keep_mask & (d <= max_radius_m)

                idx = torch.empty(B, max_polylines, dtype=torch.long)
                out_kpm = torch.ones(B, max_polylines, dtype=torch.bool)
                for b in range(B):
                    dist = d[b].clone()
                    dist[~keep_mask[b]] = float("inf")
                    order = torch.argsort(dist)
                    sel = order[:max_polylines]
                    valid_sel = torch.isfinite(dist[sel])
                    idx[b] = sel
                    out_kpm[b] = ~valid_sel

                P_ = int(roadgraph.shape[2])
                C_ = int(roadgraph.shape[3])
                idx_rg = idx[:, :, None, None].expand(B, max_polylines, P_, C_)
                rg_sel = torch.gather(roadgraph, dim=1, index=idx_rg)
                idx_pm = idx[:, :, None].expand(B, max_polylines, P_)
                pm_sel = torch.gather(padding_mask, dim=1, index=idx_pm)
                idx_mt = idx[:, :, None].expand(B, max_polylines, int(map_tokens.shape[-1]))
                mt_sel = torch.gather(map_tokens, dim=1, index=idx_mt)

                # Overwrite padding for dropped entries.
                if out_kpm.any():
                    rg_sel[out_kpm[:, :, None, None].expand_as(rg_sel)] = 0.0
                    pm_sel[out_kpm[:, :, None].expand_as(pm_sel)] = True
                    mt_sel[out_kpm[:, :, None].expand_as(mt_sel)] = 0.0

                base["roadgraph_static"] = rg_sel
                base["padding_mask_static"] = pm_sel
                base["map_tokens"] = mt_sel
                base["map_key_padding_mask"] = out_kpm
            else:
                base["map_tokens"] = map_tokens
                base["map_key_padding_mask"] = map_key_padding_mask

        # --- Traffic lights (dynamic_map_states lane_states) ---
        if any("traffic_lights" in item for item in batch):
            tl0 = None
            for item in batch:
                if "traffic_lights" in item:
                    tl0 = item["traffic_lights"]
                    break
            if tl0 is None:
                raise RuntimeError("Expected traffic_lights in batch but none found.")
            tl0 = torch.as_tensor(tl0, dtype=torch.float32)
            if tl0.dim() != 3:
                raise ValueError(f"traffic_lights must be [F,L,C], got {tuple(tl0.shape)}")
            _, L, Cd = tl0.shape

            traffic = torch.zeros(B, F, L, Cd, dtype=torch.float32)
            traffic_mask = torch.zeros(B, F, L, dtype=torch.bool)
            traffic_lane_id = torch.zeros(B, F, L, dtype=torch.long)
            has_lane_id = any("traffic_lights_lane_id" in item for item in batch)

            for b_idx, item in enumerate(batch):
                if "traffic_lights" not in item:
                    continue
                tl = torch.as_tensor(item["traffic_lights"], dtype=torch.float32)
                tl = tl[:F]
                if tl.shape[1:] != (L, Cd):
                    raise ValueError(f"traffic_lights shape mismatch: expected (*,{L},{Cd}), got {tuple(tl.shape)}")
                traffic[b_idx, : tl.shape[0]] = tl

                if "traffic_lights_mask" in item:
                    m = torch.as_tensor(item["traffic_lights_mask"], dtype=torch.bool)[:F]
                    if m.shape != (F, L):
                        # allow shorter time dimension if dataset was sliced
                        traffic_mask[b_idx, : m.shape[0]] = m
                    else:
                        traffic_mask[b_idx] = m
                else:
                    # If no mask, assume any non-zero entry is valid.
                    traffic_mask[b_idx] = (traffic[b_idx].abs().sum(dim=-1) > 0)

                if "traffic_lights_lane_id" in item:
                    lid = torch.as_tensor(item["traffic_lights_lane_id"], dtype=torch.long)[:F]
                    traffic_lane_id[b_idx, : lid.shape[0]] = lid

            # Some exports keep lane_states even when stop_point is missing (x=y=0).
            # Those coordinates are not meaningful; drop them to avoid injecting huge
            # translated tokens after ego-centering.
            xy_nonzero = traffic[:, :, :, 0:2].abs().sum(dim=-1) > 1e-6  # [B,F,L]
            traffic_mask = traffic_mask & xy_nonzero
            if has_lane_id:
                traffic_mask = traffic_mask & (traffic_lane_id != 0)

            if origin_xy is not None:
                valid = traffic_mask.unsqueeze(-1)  # [B,F,L,1]
                xy = (traffic[:, :, :, 0:2] - origin_xy) * valid.to(dtype=traffic.dtype)
                traffic[:, :, :, 0:2] = xy

            base["traffic_lights"] = traffic
            base["traffic_lights_mask"] = traffic_mask
            base["traffic_lights_lane_id"] = traffic_lane_id

            t_ref = max(0, min(int(self.obs_len) - 1, F - 1))

            # Select nearby/top-k traffic lights *before* temporal expansion.
            tl_radius_m = float(getattr(self, "tl_max_radius_m", 0.0))
            max_lights = max(1, int(getattr(self, "tl_max_lights", int(traffic.shape[2]))))
            max_lights = min(max_lights, int(traffic.shape[2]))

            base_mask = traffic_mask[:, t_ref]  # [B,L]
            base_xy = traffic[:, t_ref, :, 0:2]  # [B,L,2]
            d = torch.norm(base_xy, dim=-1)  # [B,L]
            keep_mask = base_mask
            if tl_radius_m > 0.0:
                keep_mask = keep_mask & (d <= tl_radius_m)

            idx = torch.empty(B, max_lights, dtype=torch.long)
            out_kpm = torch.ones(B, max_lights, dtype=torch.bool)
            for b in range(B):
                dist = d[b].clone()
                dist[~keep_mask[b]] = float("inf")
                order = torch.argsort(dist)
                sel = order[:max_lights]
                valid_sel = torch.isfinite(dist[sel])
                idx[b] = sel
                out_kpm[b] = ~valid_sel

            k = max(1, int(getattr(self, "tl_history_frames", 1)))
            if k <= 1:
                tl_tokens = traffic[:, t_ref]  # [B,L,Cd]
                tl_mask = base_mask  # [B,L]
                idx_tl = idx[:, :, None].expand(B, max_lights, int(tl_tokens.shape[-1]))
                tl_sel = torch.gather(tl_tokens, dim=1, index=idx_tl)
                if out_kpm.any():
                    tl_sel[out_kpm[:, :, None].expand_as(tl_sel)] = 0.0
                base["tl_tokens"] = tl_sel
                base["tl_key_padding_mask"] = out_kpm
            else:
                start = max(0, t_ref - k + 1)
                frames = torch.arange(start, t_ref + 1, device=traffic.device, dtype=torch.long)  # [Keff]
                traffic_w = traffic.index_select(dim=1, index=frames)  # [B,Keff,L,Cd]
                mask_w = traffic_mask.index_select(dim=1, index=frames)  # [B,Keff,L]

                # Keep only selected lights, then expand temporal history.
                idx_l = idx[:, None, :, None].expand(B, int(frames.numel()), max_lights, int(traffic_w.shape[-1]))
                tl_sel = torch.gather(traffic_w, dim=2, index=idx_l)  # [B,Keff,Ksel,Cd]
                idx_m = idx[:, None, :].expand(B, int(frames.numel()), max_lights)
                sel_mask = torch.gather(mask_w, dim=2, index=idx_m)  # [B,Keff,Ksel]

                # Encode time offset (0 = most recent frame = t_ref).
                offsets = (t_ref - frames).to(dtype=torch.float32)  # [Keff]
                offsets = offsets.view(1, -1, 1, 1).expand(B, -1, max_lights, 1)  # [B,Keff,Ksel,1]
                tl_sel = torch.cat([tl_sel, offsets], dim=-1)  # [B,Keff,Ksel,Cd+1]

                # Flatten time into token dimension: [B, Keff*Ksel, C]
                tl_tokens = tl_sel.reshape(B, -1, int(tl_sel.shape[-1]))
                tl_mask = sel_mask.reshape(B, -1)
                base["tl_tokens"] = tl_tokens
                base["tl_key_padding_mask"] = ~tl_mask

        return base

    def setup(self) -> None:
        if hasattr(self, "_is_setup") and self._is_setup:
            return

        dataset: Dataset = WaymoSceneDataset(
            root=self.root,
            max_agents=self.max_agents,
            num_frames=self.num_frames,
        )

        n_total = len(dataset)
        split_path = str(getattr(self, "split_path", "")).strip()
        if split_path:
            import numpy as np

            p = Path(os.path.expandvars(os.path.expanduser(split_path))).resolve()
            with np.load(p, allow_pickle=False) as d:
                train_idx = d["train_idx"].astype(np.int64)
                val_idx = d["val_idx"].astype(np.int64)
            if train_idx.size == 0 or val_idx.size == 0:
                raise ValueError(f"Split must contain non-empty train_idx and val_idx: {p}")
            if int(train_idx.max()) >= n_total or int(val_idx.max()) >= n_total:
                raise ValueError(f"Split indices out of range for dataset of length {n_total}: {p}")
            self._train_set = Subset(dataset, train_idx.tolist())
            self._val_set = Subset(dataset, val_idx.tolist())
        else:
            n_val = max(1, int(n_total * float(self.val_ratio)))
            n_train = max(1, n_total - n_val)
            if n_train + n_val > n_total:
                n_val = n_total - n_train

            gen = torch.Generator().manual_seed(int(self.seed))
            self._train_set, self._val_set = random_split(dataset, [n_train, n_val], generator=gen)
        self._is_setup = True

    def _collate(self, batch):
        base = waymo_collate_fn(batch, relative_to_ego=bool(self.relative_to_ego))

        # some extras (present only if the underlying .npz provides them)
        # Per-frame, per-agent tensors: [F, N, D]
        for key in ("velocity", "heading"):
            if not any(key in item for item in batch):
                continue
            F = base["positions"].shape[1]
            N_max = base["positions"].shape[2]
            D = None
            for item in batch:
                if key in item:
                    v = item[key]
                    if torch.is_tensor(v):
                        D = v.shape[-1] if v.dim() == 3 else 1
                    break
            if D is None:
                continue
            extra = torch.zeros(len(batch), F, N_max, D, dtype=torch.float32)
            for b_idx, item in enumerate(batch):
                if key not in item:
                    continue
                v = item[key]
                v = torch.as_tensor(v, dtype=torch.float32)
                if v.dim() == 2:
                    v = v.unsqueeze(-1)
                _, N_i, _ = v.shape
                extra[b_idx, :, :N_i, :] = v
            base[key] = extra

        # Per-agent categorical: [N]
        if any("agent_type" in item for item in batch):
            N_max = base["positions"].shape[2]
            out = torch.zeros(len(batch), N_max, dtype=torch.long)
            for b_idx, item in enumerate(batch):
                if "agent_type" not in item:
                    continue
                t = torch.as_tensor(item["agent_type"], dtype=torch.long)
                N_i = t.shape[0]
                out[b_idx, :N_i] = t
            base["agent_type"] = out

        return self._add_map_and_signal_tokens(base, batch)

    def train_dataloader(self) -> DataLoader:
        self.setup()
        return DataLoader(
            self._train_set,
            batch_size=int(self.batch_size),
            shuffle=True,
            num_workers=int(self.num_workers),
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> DataLoader:
        self.setup()
        return DataLoader(
            self._val_set,
            batch_size=int(self.batch_size),
            shuffle=False,
            num_workers=int(self.num_workers),
            collate_fn=self._collate,
        )


__all__ = ["WaymoSceneDataModule"]
