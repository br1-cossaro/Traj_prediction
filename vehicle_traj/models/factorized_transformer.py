from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .encodings import HeadingProjection
from .vehicle_transformer import MultiModeDecoder, VehicleTrajectoryTransformerOutput


def _ensure_not_all_masked(mask: torch.Tensor) -> torch.Tensor:
    """
    MultiheadAttention uses `key_padding_mask` with True = masked.
    If a full sequence is masked, some kernels can become unstable.
    We unmask the first token in those sequences.
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be [B,S], got {tuple(mask.shape)}")
    if mask.numel() == 0:
        return mask
    all_masked = mask.all(dim=1)
    if not all_masked.any():
        return mask
    mask = mask.clone()
    mask[all_masked, 0] = False
    return mask


class FactorizedEncoderLayer(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.time_attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,  # [B,S,D]
        )
        self.agent_attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )

        self.ln_time = nn.LayerNorm(int(d_model))
        self.ln_agent = nn.LayerNorm(int(d_model))
        self.ln_ffn = nn.LayerNorm(int(d_model))

        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(dim_feedforward)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(dim_feedforward), int(d_model)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, *, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: [B,F,N,D]
        valid_mask: [B,F,N] (True=valid) or None
        """
        if x.dim() != 4:
            raise ValueError(f"x must be [B,F,N,D], got {tuple(x.shape)}")
        B, F, N, D = x.shape

        # 1) Temporal attention within each agent (sequence length = F)
        y = self.ln_time(x)
        y_t = y.permute(0, 2, 1, 3).contiguous().view(B * N, F, D)  # [B*N,F,D]

        kpm_t = None
        if valid_mask is not None:
            if valid_mask.shape != (B, F, N):
                raise ValueError(f"valid_mask must be [B,F,N], got {tuple(valid_mask.shape)}")
            valid_t = valid_mask.permute(0, 2, 1).contiguous().view(B * N, F).to(dtype=torch.bool)
            kpm_t = _ensure_not_all_masked(~valid_t)

        attn_t, _ = self.time_attn(y_t, y_t, y_t, key_padding_mask=kpm_t, need_weights=False)
        attn_t = attn_t.view(B, N, F, D).permute(0, 2, 1, 3).contiguous()  # [B,F,N,D]
        x = x + self.dropout(attn_t)

        # 2) Agent attention within each timestep (sequence length = N)
        y = self.ln_agent(x)
        y_a = y.contiguous().view(B * F, N, D)  # [B*F,N,D]

        kpm_a = None
        if valid_mask is not None:
            valid_a = valid_mask.contiguous().view(B * F, N).to(dtype=torch.bool)
            kpm_a = _ensure_not_all_masked(~valid_a)

        attn_a, _ = self.agent_attn(y_a, y_a, y_a, key_padding_mask=kpm_a, need_weights=False)
        attn_a = attn_a.view(B, F, N, D)
        x = x + self.dropout(attn_a)

        # 3) FFN
        y = self.ln_ffn(x)
        x = x + self.dropout(self.ffn(y))
        return x


@dataclass(frozen=True)
class FactorizedTrajectoryTransformerConfig:
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dim_feedforward: int = 256
    max_time: int = 91
    max_agents: int = 128
    obs_len: int = 11
    num_modes: int = 1
    dropout: float = 0.1
    use_velocity: bool = False
    use_heading: bool = False
    use_type: bool = False
    type_vocab_size: int = 16
    use_map: bool = False
    use_traffic_lights: bool = False
    map_feat_dim: int = 3
    traffic_light_feat_dim: int = 3
    map_use_polyline_encoder: bool = False
    map_use_type_embedding: bool = False
    map_type_vocab_size: int = 64
    tl_use_state_embedding: bool = False
    tl_state_vocab_size: int = 16
    max_map_tokens: int = 256
    max_tl_tokens: int = 64


class FactorizedTrajectoryTransformer(nn.Module):
    """
    Factorized self-attention over (time, agents), in the spirit of Scene Transformer:
      1) time attention within each agent
      2) agent attention within each timestep

    This keeps the same output contract as `VehicleTrajectoryTransformer` so existing
    `train.py` / `evaluate.py` can be reused.
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
        map_use_polyline_encoder: bool = False,
        map_use_type_embedding: bool = False,
        map_type_vocab_size: int = 64,
        tl_use_state_embedding: bool = False,
        tl_state_vocab_size: int = 16,
        max_map_tokens: int = 256,
        max_tl_tokens: int = 64,
    ) -> None:
        super().__init__()

        self.cfg = FactorizedTrajectoryTransformerConfig(
            d_model=int(d_model),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            max_time=int(max_time),
            max_agents=int(max_agents),
            obs_len=int(obs_len),
            num_modes=int(num_modes),
            dropout=float(dropout),
            use_velocity=bool(use_velocity),
            use_heading=bool(use_heading),
            use_type=bool(use_type),
            type_vocab_size=int(type_vocab_size),
            use_map=bool(use_map),
            use_traffic_lights=bool(use_traffic_lights),
            map_feat_dim=int(map_feat_dim),
            traffic_light_feat_dim=int(traffic_light_feat_dim),
            map_use_polyline_encoder=bool(map_use_polyline_encoder),
            map_use_type_embedding=bool(map_use_type_embedding),
            map_type_vocab_size=int(map_type_vocab_size),
            tl_use_state_embedding=bool(tl_use_state_embedding),
            tl_state_vocab_size=int(tl_state_vocab_size),
            max_map_tokens=int(max_map_tokens),
            max_tl_tokens=int(max_tl_tokens),
        )

        self.pos_proj = nn.Linear(2, self.cfg.d_model)
        self.vel_proj = nn.Linear(2, self.cfg.d_model) if self.cfg.use_velocity else None
        self.heading_proj = HeadingProjection(self.cfg.d_model) if self.cfg.use_heading else None
        self.type_emb = nn.Embedding(self.cfg.type_vocab_size, self.cfg.d_model) if self.cfg.use_type else None

        self.time_emb = nn.Embedding(self.cfg.max_time, self.cfg.d_model)
        self.agent_emb = nn.Embedding(self.cfg.max_agents, self.cfg.d_model)
        self.map_slot_emb = nn.Embedding(self.cfg.max_map_tokens, self.cfg.d_model) if self.cfg.use_map else None
        self.tl_slot_emb = nn.Embedding(self.cfg.max_tl_tokens, self.cfg.d_model) if self.cfg.use_traffic_lights else None
        self.dropout = nn.Dropout(float(self.cfg.dropout))

        # Map / traffic light token encoders (mirror VehicleTrajectoryTransformer).
        self._map_proj_raw: Optional[nn.Linear] = None
        self._map_xy_proj: Optional[nn.Linear] = None
        self._map_type_emb: Optional[nn.Embedding] = None
        self._map_poly_pt_mlp: Optional[nn.Sequential] = None
        self._map_poly_fuse: Optional[nn.Linear] = None

        if self.cfg.use_map:
            if self.cfg.map_use_polyline_encoder:
                self._map_poly_pt_mlp = nn.Sequential(
                    nn.Linear(2, self.cfg.d_model),
                    nn.GELU(),
                    nn.Linear(self.cfg.d_model, self.cfg.d_model),
                )
                fuse_in = self.cfg.d_model + (self.cfg.d_model if self.cfg.map_use_type_embedding else 1)
                self._map_poly_fuse = nn.Linear(int(fuse_in), self.cfg.d_model)
                if self.cfg.map_use_type_embedding:
                    self._map_type_emb = nn.Embedding(self.cfg.map_type_vocab_size, self.cfg.d_model)
            else:
                if self.cfg.map_use_type_embedding:
                    self._map_xy_proj = nn.Linear(2, self.cfg.d_model)
                    self._map_type_emb = nn.Embedding(self.cfg.map_type_vocab_size, self.cfg.d_model)
                else:
                    self._map_proj_raw = nn.Linear(self.cfg.map_feat_dim, self.cfg.d_model)

        self._tl_proj_raw: Optional[nn.Linear] = None
        self._tl_xy_proj: Optional[nn.Linear] = None
        self._tl_state_emb: Optional[nn.Embedding] = None
        if self.cfg.use_traffic_lights:
            if self.cfg.tl_use_state_embedding:
                self._tl_xy_proj = nn.Linear(2, self.cfg.d_model)
                self._tl_state_emb = nn.Embedding(self.cfg.tl_state_vocab_size, self.cfg.d_model)
            else:
                self._tl_proj_raw = nn.Linear(self.cfg.traffic_light_feat_dim, self.cfg.d_model)

        self.layers = nn.ModuleList(
            [
                FactorizedEncoderLayer(
                    d_model=self.cfg.d_model,
                    num_heads=self.cfg.num_heads,
                    dim_feedforward=self.cfg.dim_feedforward,
                    dropout=self.cfg.dropout,
                )
                for _ in range(self.cfg.num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(self.cfg.d_model)

        self.head = nn.Linear(self.cfg.d_model, 2)
        self.mode_head = nn.Linear(self.cfg.d_model, self.cfg.num_modes) if self.cfg.num_modes > 1 else None
        self.mm_decoder = (
            MultiModeDecoder(d_model=self.cfg.d_model, num_modes=self.cfg.num_modes, dropout=self.cfg.dropout)
            if self.cfg.num_modes > 1
            else None
        )

    @staticmethod
    def _ids_from_last_channel(x: torch.Tensor, vocab_size: int) -> torch.Tensor:
        ids = x.round().to(dtype=torch.long)
        return ids.clamp(min=0, max=max(int(vocab_size) - 1, 0))

    def _encode_map_tokens(self, map_tokens: torch.Tensor) -> torch.Tensor:
        if self.cfg.map_use_polyline_encoder:
            raise RuntimeError("map_use_polyline_encoder=true: expected roadgraph_static/padding_mask_static, not map_tokens")
        if not self.cfg.map_use_type_embedding:
            if self._map_proj_raw is None:
                raise RuntimeError("map raw proj missing despite use_map=true")
            if int(map_tokens.shape[-1]) != int(self.cfg.map_feat_dim):
                raise ValueError(
                    f"map_tokens last dim must be map_feat_dim={self.cfg.map_feat_dim}, got {int(map_tokens.shape[-1])}"
                )
            return self._map_proj_raw(map_tokens)

        if int(map_tokens.shape[-1]) < 3:
            raise ValueError(f"map_tokens last dim must be >=3 when using type embedding, got {int(map_tokens.shape[-1])}")
        if self._map_xy_proj is None or self._map_type_emb is None:
            raise RuntimeError("map embedding layers missing despite map_use_type_embedding=true")
        xy = map_tokens[..., 0:2]
        type_ids = self._ids_from_last_channel(map_tokens[..., 2], self.cfg.map_type_vocab_size)
        return self._map_xy_proj(xy) + self._map_type_emb(type_ids)

    def _encode_map_polylines(
        self,
        roadgraph_static: torch.Tensor,  # [B,G,P,>=3]
        padding_mask_static: torch.Tensor,  # [B,G,P] True=pad
    ) -> torch.Tensor:
        if not self.cfg.map_use_polyline_encoder:
            raise RuntimeError("map_use_polyline_encoder=false: expected map_tokens path")
        if self._map_poly_pt_mlp is None or self._map_poly_fuse is None:
            raise RuntimeError("polyline encoder layers missing despite map_use_polyline_encoder=true")
        if roadgraph_static.dim() != 4 or int(roadgraph_static.shape[-1]) < 3:
            raise ValueError(f"roadgraph_static must be [B,G,P,>=3], got {tuple(roadgraph_static.shape)}")
        if padding_mask_static.shape != roadgraph_static.shape[:-1]:
            raise ValueError(
                f"padding_mask_static must match roadgraph_static[:-1], got {tuple(padding_mask_static.shape)} vs {tuple(roadgraph_static.shape[:-1])}"
            )

        xy = roadgraph_static[..., 0:2]  # [B,G,P,2]
        valid = (~padding_mask_static).to(dtype=xy.dtype).unsqueeze(-1)  # [B,G,P,1]
        pts = self._map_poly_pt_mlp(xy) * valid  # [B,G,P,D]
        denom = valid.sum(dim=2).clamp(min=1.0)  # [B,G,1]
        pooled = pts.sum(dim=2) / denom  # [B,G,D]

        if self.cfg.map_use_type_embedding:
            if self._map_type_emb is None:
                raise RuntimeError("map_type_emb missing despite map_use_type_embedding=true")
            type_ids = self._ids_from_last_channel(roadgraph_static[:, :, 0, 2], self.cfg.map_type_vocab_size)  # [B,G]
            type_feat = self._map_type_emb(type_ids)  # [B,G,D]
            fused_in = torch.cat([pooled, type_feat], dim=-1)  # [B,G,2D]
        else:
            type_scalar = roadgraph_static[:, :, 0, 2:3]  # [B,G,1]
            fused_in = torch.cat([pooled, type_scalar], dim=-1)  # [B,G,D+1]

        return self._map_poly_fuse(fused_in)  # [B,G,D]

    def _encode_tl_tokens(self, tl_tokens: torch.Tensor) -> torch.Tensor:
        if not self.cfg.tl_use_state_embedding:
            if self._tl_proj_raw is None:
                raise RuntimeError("traffic-light raw proj missing despite use_traffic_lights=true")
            if int(tl_tokens.shape[-1]) != int(self.cfg.traffic_light_feat_dim):
                raise ValueError(
                    f"tl_tokens last dim must be traffic_light_feat_dim={self.cfg.traffic_light_feat_dim}, got {int(tl_tokens.shape[-1])}"
                )
            return self._tl_proj_raw(tl_tokens)

        if int(tl_tokens.shape[-1]) < 3:
            raise ValueError(f"tl_tokens last dim must be >=3 when using state embedding, got {int(tl_tokens.shape[-1])}")
        if self._tl_xy_proj is None or self._tl_state_emb is None:
            raise RuntimeError("traffic-light embedding layers missing despite tl_use_state_embedding=true")
        xy = tl_tokens[..., 0:2]
        state_ids = self._ids_from_last_channel(tl_tokens[..., 2], self.cfg.tl_state_vocab_size)
        return self._tl_xy_proj(xy) + self._tl_state_emb(state_ids)

    def forward(
        self,
        *,
        positions: torch.Tensor,  # [B,F,N,2]
        valid_mask: Optional[torch.Tensor] = None,  # [B,F,N]
        velocity: Optional[torch.Tensor] = None,  # [B,F,N,2]
        heading: Optional[torch.Tensor] = None,  # [B,F,N,1] or [B,F,N]
        agent_type: Optional[torch.Tensor] = None,  # [B,N]
        map_tokens: Optional[torch.Tensor] = None,
        map_key_padding_mask: Optional[torch.Tensor] = None,
        roadgraph_static: Optional[torch.Tensor] = None,  # [B,G,P,>=3]
        padding_mask_static: Optional[torch.Tensor] = None,  # [B,G,P] True=pad
        tl_tokens: Optional[torch.Tensor] = None,
        tl_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> VehicleTrajectoryTransformerOutput:
        if positions.dim() != 4 or positions.size(-1) != 2:
            raise ValueError(f"positions must be [B,F,N,2], got {tuple(positions.shape)}")

        B, F, N, _ = positions.shape
        if N > self.cfg.max_agents:
            raise ValueError(f"N={N} exceeds max_agents={self.cfg.max_agents}")
        if F > self.cfg.max_time:
            raise ValueError(f"F={F} exceeds max_time={self.cfg.max_time}")

        x = self.pos_proj(positions)

        if self.cfg.use_velocity:
            if velocity is None:
                raise ValueError("use_velocity=true but velocity is None")
            x = x + self.vel_proj(velocity)

        if self.cfg.use_heading:
            if heading is None:
                raise ValueError("use_heading=true but heading is None")
            x = x + self.heading_proj(heading)

        if self.cfg.use_type:
            if agent_type is None:
                raise ValueError("use_type=true but agent_type is None")
            if agent_type.shape != (B, N):
                raise ValueError(f"agent_type must be [B,N], got {tuple(agent_type.shape)}")
            x = x + self.type_emb(agent_type.clamp(min=0))[:, None, :, :]

        #  append map and traffic-light tokens as "extra agents".
        map_x = None
        map_valid = None
        if self.cfg.use_map:
            if self.cfg.map_use_polyline_encoder:
                if roadgraph_static is None or padding_mask_static is None:
                    raise ValueError("use_map=true and map_use_polyline_encoder=true but roadgraph_static/padding_mask_static is missing")
                map_x = self._encode_map_polylines(roadgraph_static.to(dtype=torch.float32), padding_mask_static.to(dtype=torch.bool))
                map_valid = ~padding_mask_static.to(dtype=torch.bool).all(dim=2)  # [B,G]
            else:
                if map_tokens is None:
                    raise ValueError("use_map=true but map_tokens is None")
                if map_key_padding_mask is None:
                    raise ValueError("use_map=true but map_key_padding_mask is None")
                map_x = self._encode_map_tokens(map_tokens.to(dtype=torch.float32))
                map_valid = ~map_key_padding_mask.to(dtype=torch.bool)  # [B,G]

            if map_x.dim() != 3:
                raise ValueError(f"map tokens must be [B,G,D], got {tuple(map_x.shape)}")
            if map_valid.shape != map_x.shape[:2]:
                raise ValueError(f"map_valid must be [B,G], got {tuple(map_valid.shape)} vs {tuple(map_x.shape[:2])}")

        tl_x = None
        tl_valid = None
        if self.cfg.use_traffic_lights:
            if tl_tokens is None:
                raise ValueError("use_traffic_lights=true but tl_tokens is None")
            if tl_key_padding_mask is None:
                raise ValueError("use_traffic_lights=true but tl_key_padding_mask is None")
            tl_x = self._encode_tl_tokens(tl_tokens.to(dtype=torch.float32))
            tl_valid = ~tl_key_padding_mask.to(dtype=torch.bool)  # [B,L]
            if tl_x.dim() != 3:
                raise ValueError(f"tl tokens must be [B,L,D], got {tuple(tl_x.shape)}")
            if tl_valid.shape != tl_x.shape[:2]:
                raise ValueError(f"tl_valid must be [B,L], got {tuple(tl_valid.shape)} vs {tuple(tl_x.shape[:2])}")

        frame_idx = torch.arange(F, device=positions.device)
        x = x + self.time_emb(frame_idx)[None, :, None, :]
        x = x + self.agent_emb(torch.arange(N, device=positions.device))[None, None, :, :]

        x_total = x
        valid_total = valid_mask
        if map_x is not None:
            G = int(map_x.shape[1])
            if G > self.cfg.max_map_tokens:
                raise ValueError(f"map tokens G={G} exceeds max_map_tokens={self.cfg.max_map_tokens}")
            if self.map_slot_emb is None:
                raise RuntimeError("map_slot_emb missing despite use_map=true")
            map_seq = map_x[:, None, :, :].expand(B, F, G, self.cfg.d_model)
            map_seq = map_seq + self.time_emb(frame_idx)[None, :, None, :]
            map_seq = map_seq + self.map_slot_emb(torch.arange(G, device=positions.device))[None, None, :, :]
            x_total = torch.cat([x_total, map_seq], dim=2)
            if valid_total is not None:
                map_valid_f = map_valid[:, None, :].expand(B, F, G)
                valid_total = torch.cat([valid_total, map_valid_f], dim=2)

        if tl_x is not None:
            L = int(tl_x.shape[1])
            if L > self.cfg.max_tl_tokens:
                raise ValueError(f"traffic-light tokens L={L} exceeds max_tl_tokens={self.cfg.max_tl_tokens}")
            if self.tl_slot_emb is None:
                raise RuntimeError("tl_slot_emb missing despite use_traffic_lights=true")
            tl_seq = tl_x[:, None, :, :].expand(B, F, L, self.cfg.d_model)
            tl_seq = tl_seq + self.time_emb(frame_idx)[None, :, None, :]
            tl_seq = tl_seq + self.tl_slot_emb(torch.arange(L, device=positions.device))[None, None, :, :]
            x_total = torch.cat([x_total, tl_seq], dim=2)
            if valid_total is not None:
                tl_valid_f = tl_valid[:, None, :].expand(B, F, L)
                valid_total = torch.cat([valid_total, tl_valid_f], dim=2)

        x_total = self.dropout(x_total)

        for layer in self.layers:
            x_total = layer(x_total, valid_mask=valid_total)
        x_total = self.final_ln(x_total)

        # Slice back to agent tokens only for outputs.
        x = x_total[:, :, :N, :]

        key_padding_mask = None
        if valid_mask is not None:
            if valid_mask.shape != (B, F, N):
                raise ValueError(f"valid_mask must be [B,F,N], got {tuple(valid_mask.shape)}")
            key_padding_mask = (~valid_mask.to(dtype=torch.bool)).view(B, F * N)

        if self.cfg.num_modes <= 1 or self.mm_decoder is None or self.mode_head is None:
            deltas = self.head(x).view(B, F, N, 2)
            return VehicleTrajectoryTransformerOutput(deltas=deltas, key_padding_mask=key_padding_mask)

        # Multimodal: treat agents as A and time as T
        h = x.permute(0, 2, 1, 3).contiguous()  # [B,N,F,D]
        hM = self.mm_decoder(h)  # [B,M,N,F,D]
        deltasM = self.head(hM).permute(0, 1, 3, 2, 4).contiguous()  # [B,M,F,N,2]

        obs_idx = max(0, min(int(self.cfg.obs_len) - 1, F - 1))
        logits = self.mode_head(h[:, :, obs_idx, :])  # [B,N,M]
        return VehicleTrajectoryTransformerOutput(deltas=deltasM, key_padding_mask=key_padding_mask, mode_logits=logits)


__all__ = ["FactorizedTrajectoryTransformer", "FactorizedTrajectoryTransformerConfig"]
