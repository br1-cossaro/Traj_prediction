from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .encodings import HeadingProjection
from .vehicle_transformer import VehicleTrajectoryTransformerOutput


class _ModeConditioning(nn.Module):
    """
    Mode conditioning via  "tile + one-hot + MLP" i

    Input:
      h: [B, N, D]
    Output:
      hM: [B, M, N, D]
    """

    def __init__(self, *, d_model: int, num_modes: int, dropout: float) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.mlp = nn.Sequential(
            nn.Linear(int(d_model) + self.num_modes, int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), int(d_model)),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError(f"h must be [B,N,D], got {tuple(h.shape)}")
        B, N, D = h.shape
        M = self.num_modes
        hM = h[:, None].expand(B, M, N, D)
        onehot = torch.eye(M, device=h.device, dtype=h.dtype).view(1, M, 1, M).expand(B, M, N, M)
        x = torch.cat([hM, onehot], dim=-1)
        return self.mlp(x)


@dataclass(frozen=True)
class VehicleGRUInteractionConfig:
    d_model: int = 128
    gru_layers: int = 1

    interaction_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 256

    max_time: int = 91
    max_agents: int = 128
    obs_len: int = 11
    future_len: int = 80

    num_modes: int = 1
    dropout: float = 0.1

    use_velocity: bool = False
    use_heading: bool = False
    use_type: bool = False
    type_vocab_size: int = 16


class VehicleGRUInteraction(nn.Module):
    """
    GRU temporal encoder per agent, with an optional interaction module across agents.

    This is intended as an ablation motivated by recurrent encoders used in
    pedestrian model or multi-agent forecasting:
      - temporal inductive bias via GRU on observed frames
      - interaction modeling via a lightweight Transformer over agents at t=obs_len-1

    Output contract matches `VehicleTrajectoryTransformer` so the existing training
    and evaluation scripts can be reused.
    """

    def __init__(
        self,
        *,
        d_model: int,
        gru_layers: int,
        interaction_layers: int,
        num_heads: int,
        dim_feedforward: int,
        max_time: int,
        max_agents: int,
        obs_len: int = 11,
        future_len: int = 80,
        num_modes: int = 1,
        dropout: float = 0.1,
        use_velocity: bool = False,
        use_heading: bool = False,
        use_type: bool = False,
        type_vocab_size: int = 16,
    ) -> None:
        super().__init__()
        self.cfg = VehicleGRUInteractionConfig(
            d_model=int(d_model),
            gru_layers=int(gru_layers),
            interaction_layers=int(interaction_layers),
            num_heads=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            max_time=int(max_time),
            max_agents=int(max_agents),
            obs_len=int(obs_len),
            future_len=int(future_len),
            num_modes=int(num_modes),
            dropout=float(dropout),
            use_velocity=bool(use_velocity),
            use_heading=bool(use_heading),
            use_type=bool(use_type),
            type_vocab_size=int(type_vocab_size),
        )

        self.pos_proj = nn.Linear(2, self.cfg.d_model)
        self.vel_proj = nn.Linear(2, self.cfg.d_model) if self.cfg.use_velocity else None
        self.heading_proj = HeadingProjection(self.cfg.d_model) if self.cfg.use_heading else None
        self.type_emb = nn.Embedding(self.cfg.type_vocab_size, self.cfg.d_model) if self.cfg.use_type else None

        self.time_emb = nn.Embedding(self.cfg.max_time, self.cfg.d_model)
        self.agent_emb = nn.Embedding(self.cfg.max_agents, self.cfg.d_model)
        self.dropout = nn.Dropout(float(self.cfg.dropout))

        self.gru = nn.GRU(
            input_size=self.cfg.d_model,
            hidden_size=self.cfg.d_model,
            num_layers=self.cfg.gru_layers,
            dropout=float(self.cfg.dropout) if self.cfg.gru_layers > 1 else 0.0,
            batch_first=True,
        )

        self.interaction: Optional[nn.TransformerEncoder] = None
        if self.cfg.interaction_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=self.cfg.d_model,
                nhead=int(self.cfg.num_heads),
                dim_feedforward=int(self.cfg.dim_feedforward),
                dropout=float(self.cfg.dropout),
                activation="gelu",
                batch_first=True,  # [B, N, D]
                norm_first=True,
            )
            self.interaction = nn.TransformerEncoder(layer, num_layers=int(self.cfg.interaction_layers))

        self.future_time_emb = nn.Embedding(int(self.cfg.future_len), self.cfg.d_model)
        self.delta_head = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.d_model),
            nn.GELU(),
            nn.Linear(self.cfg.d_model, 2),
        )

        self.mode_head: Optional[nn.Linear] = None
        self.mode_cond: Optional[_ModeConditioning] = None
        if self.cfg.num_modes > 1:
            self.mode_head = nn.Linear(self.cfg.d_model, int(self.cfg.num_modes))
            self.mode_cond = _ModeConditioning(d_model=self.cfg.d_model, num_modes=int(self.cfg.num_modes), dropout=float(self.cfg.dropout))

    def forward(
        self,
        *,
        positions: torch.Tensor,  # [B,F,N,2] (normalized, with placeholders in the future)
        valid_mask: Optional[torch.Tensor] = None,  # [B,F,N]
        velocity: Optional[torch.Tensor] = None,  # [B,F,N,2]
        heading: Optional[torch.Tensor] = None,  # [B,F,N,1] or [B,F,N]
        agent_type: Optional[torch.Tensor] = None,  # [B,N]
        map_tokens: Optional[torch.Tensor] = None,
        map_key_padding_mask: Optional[torch.Tensor] = None,
        tl_tokens: Optional[torch.Tensor] = None,
        tl_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> VehicleTrajectoryTransformerOutput:
        # Map/signal support is intentionally not implemented in this ablation yet.
        _ = (map_tokens, map_key_padding_mask, tl_tokens, tl_key_padding_mask)

        if positions.dim() != 4 or positions.size(-1) != 2:
            raise ValueError(f"positions must be [B,F,N,2], got {tuple(positions.shape)}")
        B, F, N, _ = positions.shape

        if N > self.cfg.max_agents:
            raise ValueError(f"N={N} exceeds max_agents={self.cfg.max_agents}")
        if F > self.cfg.max_time:
            raise ValueError(f"F={F} exceeds max_time={self.cfg.max_time}")
        if int(self.cfg.obs_len) + int(self.cfg.future_len) > F:
            raise ValueError(
                f"Need F >= obs_len+future_len ({self.cfg.obs_len}+{self.cfg.future_len}), got F={F}"
            )

        token = self.pos_proj(positions)

        if self.cfg.use_velocity:
            if velocity is None:
                raise ValueError("use_velocity=true but velocity is None")
            token = token + self.vel_proj(velocity)

        if self.cfg.use_heading:
            if heading is None:
                raise ValueError("use_heading=true but heading is None")
            token = token + self.heading_proj(heading)

        frame_idx = torch.arange(F, device=positions.device)
        agent_idx = torch.arange(N, device=positions.device)
        token = token + self.time_emb(frame_idx)[None, :, None, :]
        token = token + self.agent_emb(agent_idx)[None, None, :, :]

        if self.cfg.use_type:
            if agent_type is None:
                raise ValueError("use_type=true but agent_type is None")
            if agent_type.shape != (B, N):
                raise ValueError(f"agent_type must be [B,N], got {tuple(agent_type.shape)}")
            token = token + self.type_emb(agent_type.clamp(min=0))[:, None, :, :]

        token = self.dropout(token)

        # Use only observed frames for the temporal encoder.
        obs_len = int(self.cfg.obs_len)
        tok_obs = token[:, :obs_len]  # [B,obs,N,D]

        # GRU per agent: treat agents as batch elements.
        x = tok_obs.permute(0, 2, 1, 3).contiguous().view(B * N, obs_len, self.cfg.d_model)  # [B*N,obs,D]
        _, h_n = self.gru(x)  # h_n: [L, B*N, D]
        h = h_n[-1].view(B, N, self.cfg.d_model)  # [B,N,D]

        # Optional interaction across agents at the last observed timestep.
        if self.interaction is not None:
            kpm_agent: Optional[torch.Tensor] = None
            if valid_mask is not None:
                if valid_mask.shape != (B, F, N):
                    raise ValueError(f"valid_mask must be [B,F,N], got {tuple(valid_mask.shape)}")
                valid_last = valid_mask[:, obs_len - 1].to(dtype=torch.bool)  # [B,N]
                kpm_agent = ~valid_last
                # If all agents are masked for a scene, unmask the first token to keep Transformer happy.
                all_masked = kpm_agent.all(dim=1)
                if all_masked.any():
                    kpm_agent = kpm_agent.clone()
                    kpm_agent[all_masked, 0] = False
            h = self.interaction(h, src_key_padding_mask=kpm_agent)

        # Decode future offsets (absolute delta to last observed position for each future timestep).
        fut_len = int(self.cfg.future_len)
        t_emb = self.future_time_emb(torch.arange(fut_len, device=positions.device))  # [Tf,D]

        if self.cfg.num_modes <= 1 or self.mode_cond is None or self.mode_head is None:
            z = h[:, None, :, :] + t_emb[None, :, None, :]  # [B,Tf,N,D]
            deltas_fut = self.delta_head(z)  # [B,Tf,N,2]
            deltas = torch.zeros(B, F, N, 2, device=positions.device, dtype=deltas_fut.dtype)
            deltas[:, obs_len : obs_len + fut_len] = deltas_fut
            kpm = (~valid_mask.to(dtype=torch.bool)).view(B, F * N) if valid_mask is not None else None
            return VehicleTrajectoryTransformerOutput(deltas=deltas, key_padding_mask=kpm)

        # Multimodal: mode-condition per agent, then decode per-mode.
        hM = self.mode_cond(h)  # [B,M,N,D]
        zM = hM[:, :, None, :, :] + t_emb[None, None, :, None, :]  # [B,M,Tf,N,D]
        deltasM_fut = self.delta_head(zM)  # [B,M,Tf,N,2]
        deltasM = torch.zeros(B, int(self.cfg.num_modes), F, N, 2, device=positions.device, dtype=deltasM_fut.dtype)
        deltasM[:, :, obs_len : obs_len + fut_len] = deltasM_fut
        logits = self.mode_head(h)  # [B,N,M]
        kpm = (~valid_mask.to(dtype=torch.bool)).view(B, F * N) if valid_mask is not None else None
        return VehicleTrajectoryTransformerOutput(deltas=deltasM, key_padding_mask=kpm, mode_logits=logits)


__all__ = ["VehicleGRUInteraction", "VehicleGRUInteractionConfig"]

