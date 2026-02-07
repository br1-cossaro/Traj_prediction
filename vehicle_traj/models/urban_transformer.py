from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .encodings import HeadingProjection
from .vehicle_transformer import (
    MultiModeDecoder,
    VehicleTrajectoryTransformerOutput,
    _ensure_not_all_masked,
)


@dataclass(frozen=True)
class UrbanVehicleTransformerOutput(VehicleTrajectoryTransformerOutput):
    """Output structure for UrbanVehicleTransformer.

    - deltas: [B,M,F,N,2] or [B,F,N,2] predicted future position deltas (zeroed for observed steps)
    - key_padding_mask: [B,F*N] boolean mask indicating valid agent-time tokens
    - mode_logits: [B,N,M] optional mode confidence logits (only if num_modes > 1)
    """

class UrbanVehicleTransformer(nn.Module):
    """Lightweight Transformer encoder for vehicle trajectories.

    - Encodes per-agent, per-timestep tokens (positions + optional kinematics/type/map features).
    - Supports optional cross-attention to map and traffic-light tokens.
    - Produces multi-modal future deltas matching the vehicle_traj training pipeline.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        max_time: int,
        max_agents: int,
        obs_len: int = 11,
        num_modes: int = 1,
        dropout: float = 0.1,
        use_velocity: bool = False,
        use_heading: bool = False,
        use_type: bool = False,
        type_vocab_size: int = 16,
        use_map: bool = False,
        use_traffic_lights: bool = False,
        map_feat_dim: int = 3,
        traffic_light_feat_dim: int = 3,
        map_integration: str = "none",
        tl_integration: str = "none",
        use_agent_map_feat: bool = False,
        agent_map_feat_dim: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.obs_len = int(obs_len)
        self.num_modes = int(num_modes)
        self.use_velocity = bool(use_velocity)
        self.use_heading = bool(use_heading)
        self.use_type = bool(use_type)
        self.use_map = bool(use_map)
        self.use_traffic_lights = bool(use_traffic_lights)
        self.use_agent_map_feat = bool(use_agent_map_feat)

        valid_integration = {"none", "cross_attn"}
        if map_integration not in valid_integration:
            raise ValueError(f"map_integration must be one of {sorted(valid_integration)}")
        if tl_integration not in valid_integration:
            raise ValueError(f"tl_integration must be one of {sorted(valid_integration)}")
        self.map_integration = map_integration
        self.tl_integration = tl_integration

        self.pos_proj = nn.Linear(2, self.d_model)
        self.vel_proj = nn.Linear(2, self.d_model) if self.use_velocity else None
        self.heading_proj = HeadingProjection(self.d_model) if self.use_heading else None
        self.type_emb = nn.Embedding(int(type_vocab_size), self.d_model) if self.use_type else None
        self.agent_map_proj = (
            nn.Linear(int(agent_map_feat_dim), self.d_model) if self.use_agent_map_feat else None
        )

        self.time_emb = nn.Embedding(int(max_time), self.d_model)
        self.agent_emb = nn.Embedding(int(max_agents), self.d_model)
        self.dropout = nn.Dropout(float(dropout))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.map_proj = nn.Linear(int(map_feat_dim), self.d_model) if self.use_map else None
        self.tl_proj = nn.Linear(int(traffic_light_feat_dim), self.d_model) if self.use_traffic_lights else None
        self.map_attn = (
            nn.MultiheadAttention(self.d_model, int(num_heads), dropout=float(dropout), batch_first=True)
            if self.use_map and self.map_integration == "cross_attn"
            else None
        )
        self.tl_attn = (
            nn.MultiheadAttention(self.d_model, int(num_heads), dropout=float(dropout), batch_first=True)
            if self.use_traffic_lights and self.tl_integration == "cross_attn"
            else None
        )

        self.modes = MultiModeDecoder(d_model=self.d_model, num_modes=self.num_modes, dropout=float(dropout))
        self.delta_head = nn.Linear(self.d_model, 2)
        self.mode_head = nn.Linear(self.d_model, self.num_modes) if self.num_modes > 1 else None

    def _encode_tokens(
        self,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        heading: Optional[torch.Tensor] = None,
        agent_type: Optional[torch.Tensor] = None,
        agent_map_feat: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed agent-time tokens and return (tokens, key_padding_mask)."""

        B, F, N, _ = positions.shape
        tok = self.pos_proj(positions)
        if self.use_velocity:
            if velocity is None:
                raise ValueError("use_velocity=true but velocity is None")
            tok = tok + self.vel_proj(velocity)
        if self.use_heading:
            if heading is None:
                raise ValueError("use_heading=true but heading is None")
            tok = tok + self.heading_proj(heading)
        if self.use_type:
            if agent_type is None:
                raise ValueError("use_type=true but agent_type is None")
            type_emb = self.type_emb(agent_type).view(B, 1, N, -1)
            tok = tok + type_emb
        if self.use_agent_map_feat:
            if agent_map_feat is None:
                raise ValueError("use_agent_map_feat=true but agent_map_feat is None")
            tok = tok + self.agent_map_proj(agent_map_feat)

        t_ids = torch.arange(F, device=positions.device)
        a_ids = torch.arange(N, device=positions.device)
        time_emb = self.time_emb(t_ids).view(1, F, 1, -1)       # [1,F,1,D]
        agent_emb = self.agent_emb(a_ids).view(1, 1, N, -1)     # [1,1,N,D]
        tok = tok + time_emb + agent_emb
        tok = self.dropout(tok)

        tokens = tok.view(B, F * N, self.d_model)
        key_padding_mask = ~valid_mask.view(B, F * N)
        key_padding_mask = _ensure_not_all_masked(key_padding_mask)
        return tokens, key_padding_mask

    def _ctx_attend(
        self,
        x: torch.Tensor,
        ctx: Optional[torch.Tensor],
        kpm: Optional[torch.Tensor],
        attn: Optional[nn.MultiheadAttention],
    ) -> torch.Tensor:
        if ctx is None or attn is None:
            return x
        kpm = _ensure_not_all_masked(kpm) if kpm is not None else None
        ctx_out, _ = attn(x, ctx, ctx, key_padding_mask=kpm, need_weights=False)
        return x + self.dropout(ctx_out)

    def forward(
        self,
        *,
        positions: torch.Tensor,  # [B,F,N,2]
        valid_mask: torch.Tensor,  # [B,F,N]
        velocity: Optional[torch.Tensor] = None,
        heading: Optional[torch.Tensor] = None,
        agent_type: Optional[torch.Tensor] = None,
        agent_map_feat: Optional[torch.Tensor] = None,
        map_tokens: Optional[torch.Tensor] = None,  # [B,G,feat]
        map_key_padding_mask: Optional[torch.Tensor] = None,  # [B,G]
        roadgraph_static: Optional[torch.Tensor] = None,  # unused, kept for API parity
        padding_mask_static: Optional[torch.Tensor] = None,  # unused
        tl_tokens: Optional[torch.Tensor] = None,  # [B,L,feat]
        tl_key_padding_mask: Optional[torch.Tensor] = None,  # [B,L]
    ) -> UrbanVehicleTransformerOutput:
        tokens, key_padding_mask = self._encode_tokens(
            positions,
            valid_mask,
            velocity=velocity,
            heading=heading,
            agent_type=agent_type,
            agent_map_feat=agent_map_feat,
        )

        x = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        if self.use_map and map_tokens is not None and self.map_proj is not None and self.map_attn is not None:
            ctx = self.map_proj(map_tokens)
            x = self._ctx_attend(x, ctx, map_key_padding_mask, self.map_attn)
        if self.use_traffic_lights and tl_tokens is not None and self.tl_proj is not None and self.tl_attn is not None:
            ctx = self.tl_proj(tl_tokens)
            x = self._ctx_attend(x, ctx, tl_key_padding_mask, self.tl_attn)

        B, F, N, D = positions.shape[0], positions.shape[1], positions.shape[2], self.d_model
        x_tok = x.view(B, F, N, D)

        if self.num_modes > 1:
            agent_first = x_tok.permute(0, 2, 1, 3)  # [B,N,F,D]
            decoded = self.modes(agent_first)  # [B,M,N,F,D]
            deltas = self.delta_head(decoded).permute(0, 1, 3, 2, 4)  # [B,M,F,N,2]
        else:
            deltas = self.delta_head(x_tok)  # [B,F,N,2]

        # Zero out observed steps to match training target convention.
        if deltas.dim() == 5:
            deltas[:, :, : self.obs_len] = 0.0
        else:
            deltas[:, : self.obs_len] = 0.0

        mode_logits = None
        if self.num_modes > 1:
            last_obs = x_tok[:, self.obs_len - 1]  # [B,N,D]
            mode_logits = self.mode_head(last_obs)

        return UrbanVehicleTransformerOutput(
            deltas=deltas,
            key_padding_mask=key_padding_mask,
            mode_logits=mode_logits,
        )


__all__ = ["UrbanVehicleTransformer", "UrbanVehicleTransformerOutput"]
