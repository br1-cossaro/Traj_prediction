"""
Waymo multi-agent scene dataset and collate function.

Design:
    - Scene-level samples with multiple agents.
    - The dataset reads preprocessed .npz files that contain:
        positions:   [F, N, 2]  (x, y in global coordinates)
        valid_mask:  [F, N]     (bool, True if the agent is present and valid)
        ego_idx:     ()         (int, index of the reference agent)

    - On Grid, a preprocessing script should generate these .npz files
      from Waymo TFRecords.

    - __getitem__(idx) returns a dict for one scene:
        {
            "positions":  Tensor[F, N_i, 2],
            "valid_mask": Tensor[F, N_i],
            "ego_idx":    int,
            "scenario_id": str (optional)
        }

    - collate_fn builds a batch:
        {
            "positions":  Tensor[B, F, N_max, 2],
            "valid_mask": Tensor[B, F, N_max],
            "ego_idx":    Tensor[B],
            "scenario_id": list[str]
        }

      N_max is the max number of agents in the batch. Smaller scenes are padded
      with valid_mask=False.

    - Optionally, collate_fn converts positions to coordinates relative to the
      ego position at the first valid frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WaymoSceneSample:
    """
    Single multi-agent scene sample.

    Used for clarity and type checking.
    """

    positions: torch.Tensor  # [F, N_i, 2]
    valid_mask: torch.Tensor  # [F, N_i]
    ego_idx: int
    scenario_id: str


class WaymoSceneDataset(Dataset):
    """
    Waymo multi-agent scene dataset.

    This dataset does not read Waymo TFRecords directly. It expects a
    preprocessing step that generates .npz files with the keys:
        - positions:  [F, N, 2]
        - valid_mask: [F, N]
        - ego_idx:    scalar int

    How its used normally:
        dataset = WaymoSceneDataset(
            root='data/processed/waymo_scenes_train',
            max_agents=32,
            num_frames=91,
        )
    """

    def __init__(
        self,
        root: str | Path,
        max_agents: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> None:
        """
        Args:
            root:
                Directory with .npz files (one per scene) or a single .npz
                file that contains multiple scenes.
            max_agents:
                Max number of agents to keep per scene.
                If None, keep all agents.
            num_frames:
                If set, truncate (or validate) the number of frames F.
                Typical for Waymo: 91 (10 past + 1 present + 80 future).
        """
        super().__init__()
        self.root = Path(root).expanduser()
        self.max_agents = max_agents
        self.num_frames = num_frames

        # Index mapping so we can support:
        #   (a) one file per scene: positions [F,N,2]
        #   (b) one file with many scenes: positions [S,F,N,2]
        #   (c) a directory of shard files (each a multi-scene .npz)
        self._index: List[Tuple[Path, Optional[int]]] = []

        if self.root.is_dir():
            files: List[Path] = sorted(p for p in self.root.glob("*.npz") if p.is_file())
            if not files:
                raise FileNotFoundError(f"No .npz files found in {self.root}")
            for path in files:
                with np.load(path, allow_pickle=True) as d:
                    if "positions" not in d:
                        raise KeyError(f"Expected 'positions' in npz file {path}")
                    positions = d["positions"]
                    if positions.ndim == 4:
                        S = int(positions.shape[0])
                        self._index.extend((path, i) for i in range(S))
                    elif positions.ndim == 3:
                        self._index.append((path, None))
                    else:
                        raise ValueError(
                            f"Expected positions [F,N,2] or [S,F,N,2] in {path}, "
                            f"got {positions.shape}"
                        )
        else:
            if self.root.suffix != ".npz":
                raise ValueError(f"Path {self.root} is not a directory or a .npz")
            with np.load(self.root, allow_pickle=True) as d:
                if "positions" not in d:
                    raise KeyError("Expected a 'positions' key in npz file " f"{self.root}")
                positions = d["positions"]
                if positions.ndim == 4:
                    S = int(positions.shape[0])
                    self._index.extend((self.root, i) for i in range(S))
                elif positions.ndim == 3:
                    self._index.append((self.root, None))
                else:
                    raise ValueError(
                        "Expected positions with shape [F,N,2] or [S,F,N,2]; "
                        f"got {positions.shape}"
                    )

        self.num_scenes = len(self._index)

    def __len__(self) -> int:
        return self.num_scenes

    def _decode_scenario_id(self, x: Any) -> str:
        if isinstance(x, bytes):
            try:
                return x.decode("utf-8", errors="replace")
            except Exception:
                return str(x)
        return str(x)

    def _load_scene_from_npz(self, path: Path, scene_idx: Optional[int]) -> Dict[str, Any]:
        """
        Loads either:
          - a single-scene npz: positions [F,N,2]
          - a multi-scene npz: positions [S,F,N,2] (scene_idx selects along S)
        """
        extras: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=True) as d:
            if "positions" not in d or "valid_mask" not in d:
                raise KeyError(f"File {path} must contain 'positions' and 'valid_mask'")

            positions_all = d["positions"]
            valid_all = d["valid_mask"]

            if positions_all.ndim == 4:
                if scene_idx is None:
                    raise ValueError(f"scene_idx is required for a multi-scene npz: {path}")
                positions = positions_all[scene_idx]  # [F,N,2]
                valid_mask = valid_all[scene_idx]  # [F,N]
                ego_idx = int(d["ego_idx"][scene_idx]) if "ego_idx" in d else 0

                scenario_id = f"{path.stem}_{scene_idx}"
                if "scenario_id" in d:
                    try:
                        scenario_id = self._decode_scenario_id(d["scenario_id"][scene_idx])
                    except Exception:
                        pass

                # some other options (support both `velocity` and WOMD-style velocities)
                if "velocity" in d:
                    extras["velocity"] = d["velocity"][scene_idx]
                elif "velocities" in d:
                    extras["velocity"] = d["velocities"][scene_idx]
                if "heading" in d:
                    extras["heading"] = d["heading"][scene_idx]
                if "agent_type" in d:
                    extras["agent_type"] = d["agent_type"][scene_idx]

                # Optional map + signals (static roadgraph, traffic lights, dynamic lanes)
                if "roadgraph_static" in d:
                    extras["roadgraph_static"] = d["roadgraph_static"][scene_idx]
                if "roadgraph_static_mask" in d:
                    extras["roadgraph_static_mask"] = d["roadgraph_static_mask"][scene_idx]
                if "padding_mask_static" in d:
                    extras["padding_mask_static"] = d["padding_mask_static"][scene_idx]

                if "traffic_lights" in d:
                    extras["traffic_lights"] = d["traffic_lights"][scene_idx]
                if "traffic_lights_mask" in d:
                    extras["traffic_lights_mask"] = d["traffic_lights_mask"][scene_idx]
                if "traffic_lights_lane_id" in d:
                    extras["traffic_lights_lane_id"] = d["traffic_lights_lane_id"][scene_idx]

                if "roadgraph_dynamic" in d:
                    extras["roadgraph_dynamic"] = d["roadgraph_dynamic"][scene_idx]
                if "padding_mask_dynamic" in d:
                    extras["padding_mask_dynamic"] = d["padding_mask_dynamic"][scene_idx]
                if "roadgraph_dynamic_lane_id" in d:
                    extras["roadgraph_dynamic_lane_id"] = d["roadgraph_dynamic_lane_id"][scene_idx]
            elif positions_all.ndim == 3:
                positions = positions_all  # [F,N,2]
                valid_mask = valid_all  # [F,N]
                ego_idx = int(d["ego_idx"]) if "ego_idx" in d else 0

                scenario_id = path.stem
                if "scenario_id" in d:
                    try:
                        scenario_id = self._decode_scenario_id(d["scenario_id"])
                    except Exception:
                        pass

                if "velocity" in d:
                    extras["velocity"] = d["velocity"]
                elif "velocities" in d:
                    extras["velocity"] = d["velocities"]
                if "heading" in d:
                    extras["heading"] = d["heading"]
                if "agent_type" in d:
                    extras["agent_type"] = d["agent_type"]

                if "roadgraph_static" in d:
                    extras["roadgraph_static"] = d["roadgraph_static"]
                if "roadgraph_static_mask" in d:
                    extras["roadgraph_static_mask"] = d["roadgraph_static_mask"]
                if "padding_mask_static" in d:
                    extras["padding_mask_static"] = d["padding_mask_static"]

                if "traffic_lights" in d:
                    extras["traffic_lights"] = d["traffic_lights"]
                if "traffic_lights_mask" in d:
                    extras["traffic_lights_mask"] = d["traffic_lights_mask"]
                if "traffic_lights_lane_id" in d:
                    extras["traffic_lights_lane_id"] = d["traffic_lights_lane_id"]

                if "roadgraph_dynamic" in d:
                    extras["roadgraph_dynamic"] = d["roadgraph_dynamic"]
                if "padding_mask_dynamic" in d:
                    extras["padding_mask_dynamic"] = d["padding_mask_dynamic"]
                if "roadgraph_dynamic_lane_id" in d:
                    extras["roadgraph_dynamic_lane_id"] = d["roadgraph_dynamic_lane_id"]
            else:
                raise ValueError(f"positions has unexpected shape in {path}: {positions_all.shape}")

        sample = self._build_scene_sample(
            positions=positions,
            valid_mask=valid_mask,
            ego_idx=ego_idx,
            scenario_id=scenario_id,
        )
        sample_dict: Dict[str, Any] = {
            "positions": sample.positions,
            "valid_mask": sample.valid_mask,
            "ego_idx": sample.ego_idx,
            "scenario_id": sample.scenario_id,
        }
        sample_dict.update(
            self._build_extras(
                extras,
                scenario_id=sample.scenario_id,
                F=sample.positions.shape[0],
                N=sample.positions.shape[1],
            )
        )
        return sample_dict

    def _build_extras(self, extras: dict[str, np.ndarray], scenario_id: str, F: int, N: int) -> Dict[str, Any]:
        """
        Applies the same temporal/agent truncations to optional arrays and converts to tensors.
        """
        out: Dict[str, Any] = {}

        def _slice_time_agents(x: np.ndarray) -> np.ndarray:
            if x.ndim >= 2 and x.shape[0] >= F:
                x = x[:F]
            if x.ndim >= 2 and x.shape[1] >= N:
                x = x[:, :N]
            return x

        if "velocity" in extras:
            v = extras["velocity"]
            if v.ndim != 3 or v.shape[-1] != 2:
                raise ValueError(f"velocity must be [F,N,2] for {scenario_id}, got {v.shape}")
            v = _slice_time_agents(v)
            out["velocity"] = torch.as_tensor(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)

        if "heading" in extras:
            h = extras["heading"]
            if h.ndim == 2:
                h = h[:, :, None]
            if h.ndim != 3 or h.shape[-1] not in (1, 2):
                raise ValueError(f"heading must be [F,N] or [F,N,1|2] for {scenario_id}, got {h.shape}")
            h = _slice_time_agents(h)
            out["heading"] = torch.as_tensor(np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)

        if "agent_type" in extras:
            t = extras["agent_type"]
            if t.ndim != 1:
                raise ValueError(f"agent_type must be [N] for {scenario_id}, got {t.shape}")
            t = t[:N]
            out["agent_type"] = torch.as_tensor(t, dtype=torch.long)

        if "roadgraph_static" in extras:
            rg = extras["roadgraph_static"]
            if rg.ndim != 3 or rg.shape[-1] < 2:
                raise ValueError(f"roadgraph_static must be [Gs,P,C>=2] for {scenario_id}, got {rg.shape}")
            out["roadgraph_static"] = torch.as_tensor(np.nan_to_num(rg, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)

        if "padding_mask_static" in extras:
            pm = extras["padding_mask_static"]
            if pm.ndim != 2:
                raise ValueError(f"padding_mask_static must be [Gs,P] for {scenario_id}, got {pm.shape}")
            out["padding_mask_static"] = torch.as_tensor(pm, dtype=torch.bool)
        elif "roadgraph_static_mask" in extras:
            m = extras["roadgraph_static_mask"]
            if m.ndim != 2:
                raise ValueError(f"roadgraph_static_mask must be [Gs,P] for {scenario_id}, got {m.shape}")
            out["roadgraph_static_mask"] = torch.as_tensor(m, dtype=torch.bool)

        if "traffic_lights" in extras:
            tl = extras["traffic_lights"]
            if tl.ndim != 3 or tl.shape[-1] < 2:
                raise ValueError(f"traffic_lights must be [F,L,C>=2] for {scenario_id}, got {tl.shape}")
            tl = tl[:F]
            out["traffic_lights"] = torch.as_tensor(np.nan_to_num(tl, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)

        if "traffic_lights_mask" in extras:
            tlm = extras["traffic_lights_mask"]
            if tlm.ndim != 2:
                raise ValueError(f"traffic_lights_mask must be [F,L] for {scenario_id}, got {tlm.shape}")
            tlm = tlm[:F]
            out["traffic_lights_mask"] = torch.as_tensor(tlm, dtype=torch.bool)

        if "traffic_lights_lane_id" in extras:
            lid = extras["traffic_lights_lane_id"]
            if lid.ndim != 2:
                raise ValueError(f"traffic_lights_lane_id must be [F,L] for {scenario_id}, got {lid.shape}")
            lid = lid[:F]
            out["traffic_lights_lane_id"] = torch.as_tensor(lid, dtype=torch.long)

        if "roadgraph_dynamic" in extras:
            rgd = extras["roadgraph_dynamic"]
            if rgd.ndim != 3 or rgd.shape[-1] < 2:
                raise ValueError(f"roadgraph_dynamic must be [Gd,T,C>=2] for {scenario_id}, got {rgd.shape}")
            # Keep full time dimension (T) here; downstream may slice to obs_len or match F.
            out["roadgraph_dynamic"] = torch.as_tensor(np.nan_to_num(rgd, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)

        if "padding_mask_dynamic" in extras:
            pmd = extras["padding_mask_dynamic"]
            if pmd.ndim != 2:
                raise ValueError(f"padding_mask_dynamic must be [Gd,T] for {scenario_id}, got {pmd.shape}")
            out["padding_mask_dynamic"] = torch.as_tensor(pmd, dtype=torch.bool)

        if "roadgraph_dynamic_lane_id" in extras:
            rgd_id = extras["roadgraph_dynamic_lane_id"]
            if rgd_id.ndim != 1:
                raise ValueError(f"roadgraph_dynamic_lane_id must be [Gd] for {scenario_id}, got {rgd_id.shape}")
            out["roadgraph_dynamic_lane_id"] = torch.as_tensor(rgd_id, dtype=torch.long)

        return out

    def _build_scene_sample(
        self,
        positions: np.ndarray,
        valid_mask: np.ndarray,
        ego_idx: int,
        scenario_id: str,
    ) -> WaymoSceneSample:
        """
        Apply time/agent truncation and convert to tensors.
        """
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError(
                "Expected positions with shape [F, N, 2]; "
                f"got {positions.shape}"
            )
        if valid_mask.shape != positions.shape[:2]:
            raise ValueError(
                "valid_mask must match positions shape [F, N]; "
                f"valid_mask={valid_mask.shape}, positions={positions.shape}"
            )

        F, N, _ = positions.shape

        # Temporal trim if num_frames is set
        if self.num_frames is not None:
            if F < self.num_frames:
                raise ValueError(
                    f"Scene {scenario_id} has only {F} frames, "
                    f"expected at least {self.num_frames}"
                )
            positions = positions[: self.num_frames]
            valid_mask = valid_mask[: self.num_frames]
            F = self.num_frames

        # Limit number of agents if max_agents is set
        if self.max_agents is not None and N > self.max_agents:
            # Simple strategy: keep the first max_agents.
            # Later we can sort by valid frame count, etc.
            positions = positions[:, : self.max_agents, :]
            valid_mask = valid_mask[:, : self.max_agents]
            N = self.max_agents

        # Ensure ego_idx is in [0, N)
        if ego_idx < 0 or ego_idx >= N:
            ego_idx = 0

        # Convert to PyTorch tensors
        positions_t = torch.as_tensor(positions, dtype=torch.float32)  # [F, N, 2]
        valid_mask_t = torch.as_tensor(valid_mask, dtype=torch.bool)  # [F, N]
        # Replace NaNs/Infs and zero-out invalid positions
        positions_t = torch.nan_to_num(positions_t, nan=0.0, posinf=0.0, neginf=0.0)
        positions_t = positions_t * valid_mask_t.unsqueeze(-1)

        return WaymoSceneSample(
            positions=positions_t,
            valid_mask=valid_mask_t,
            ego_idx=ego_idx,
            scenario_id=scenario_id,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, scene_idx = self._index[idx]
        return self._load_scene_from_npz(path, scene_idx)


def waymo_collate_fn(
    batch: Sequence[Dict[str, Any]],
    relative_to_ego: bool = True,
) -> Dict[str, Any]:
    """
    Collate function for Waymo scenes.

    Takes a list of samples (each with shape [F, N_i, 2]) and builds batch
    tensors:

        positions:  [B, F, N_max, 2]
        valid_mask: [B, F, N_max]
        ego_idx:    [B]

    N_max = max_i N_i in the batch. Extra slots are padded with zeros and
    valid_mask=False.

    If relative_to_ego=True, positions are converted to coordinates relative
    to the ego position at the first valid frame.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    # Check all scenes have the same number of frames
    F_list = [item["positions"].shape[0] for item in batch]
    if len(set(F_list)) != 1:
        raise ValueError(
            "All scenes in the batch must have the same number of frames. "
            f"Got: {F_list}"
        )

    B = len(batch)
    F = F_list[0]
    N_list = [item["positions"].shape[1] for item in batch]
    N_max = max(N_list)

    positions_batch = torch.zeros(B, F, N_max, 2, dtype=torch.float32)
    valid_mask_batch = torch.zeros(B, F, N_max, dtype=torch.bool)
    ego_indices = torch.zeros(B, dtype=torch.long)
    ego_origin = torch.zeros(B, 2, dtype=torch.float32)
    ego_frame_ref = torch.zeros(B, dtype=torch.long)
    scenario_ids: List[str] = []

    for b_idx, item in enumerate(batch):
        pos = item["positions"]  # [F, N_i, 2]
        mask = item["valid_mask"]  # [F, N_i]
        ego_idx = int(item.get("ego_idx", 0))
        scenario_id = str(item.get("scenario_id", f"scene_{b_idx}"))

        F_i, N_i, _ = pos.shape
        if F_i != F:
            raise ValueError(
                "Frame count mismatch inside batch: "
                f"expected {F}, got {F_i}"
            )

        positions_batch[b_idx, :, :N_i, :] = pos
        valid_mask_batch[b_idx, :, :N_i] = mask
        ego_indices[b_idx] = ego_idx if 0 <= ego_idx < N_i else 0
        scenario_ids.append(scenario_id)

    if relative_to_ego:
        # Convert to ego-relative coordinates.
        # For each scene b:
        #   - find the first frame where ego is valid
        #   - subtract its position from all agents in all frames
        for b_idx in range(B):
            ego_idx = int(ego_indices[b_idx].item())
            ego_valid = valid_mask_batch[b_idx, :, ego_idx]  # [F]

            if ego_valid.any():
                frame_ref = int(torch.nonzero(ego_valid, as_tuple=False)[0].item())
            else:
                # If ego is never valid, use frame 0 as fallback
                frame_ref = 0

            ego_pos_ref = positions_batch[b_idx, frame_ref, ego_idx, :]  # [2]
            ego_origin[b_idx] = ego_pos_ref
            ego_frame_ref[b_idx] = frame_ref
            # Avoid in-place aliasing issues by reassigning
            positions_batch[b_idx] = positions_batch[b_idx] - ego_pos_ref.view(1, 1, 2)
    else:
        # Still provide a consistent origin reference (frame 0) for downstream consumers.
        for b_idx in range(B):
            ego_idx = int(ego_indices[b_idx].item())
            ego_origin[b_idx] = positions_batch[b_idx, 0, ego_idx, :]
            ego_frame_ref[b_idx] = 0

    return {
        "positions": positions_batch,  # [B, F, N_max, 2] relative or absolute
        "valid_mask": valid_mask_batch,  # [B, F, N_max]
        "ego_idx": ego_indices,  # [B]
        "ego_origin": ego_origin,  # [B,2] (absolute if relative_to_ego=False, else reference used for translation)
        "ego_frame_ref": ego_frame_ref,  # [B]
        "scenario_id": scenario_ids,
    }


__all__ = ["WaymoSceneDataset", "waymo_collate_fn"]
