from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .encodings import HeadingProjection


def _ensure_not_all_masked(mask: torch.Tensor) -> torch.Tensor:
    """
    MultiheadAttention uses `key_padding_mask` with True = masked.
    If an entire sequence is masked, some kernels can become unstable.
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


class MultiModeDecoder(nn.Module):
    """
    multi-future tiling :
      [B,A,T,D] -> tile M -> concat one-hot(M) -> MLP -> [B,M,A,T,D]
    """

    def __init__(self, *, d_model: int, num_modes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.mlp = nn.Sequential(
            nn.Linear(int(d_model) + self.num_modes, int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), int(d_model)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,A,T,D]
        if x.dim() != 4:
            raise ValueError(f"x must be [B,A,T,D], got {tuple(x.shape)}")
        B, A, T, D = x.shape
        M = self.num_modes
        xM = x[:, None].expand(B, M, A, T, D)
        onehot = torch.eye(M, device=x.device, dtype=x.dtype).view(1, M, 1, 1, M).expand(B, M, A, T, M)
        h = torch.cat([xM, onehot], dim=-1)
        return self.mlp(h)  # [B,M,A,T,D]


@dataclass(frozen=True)
class VehicleTrajectoryTransformerOutput:
    deltas: torch.Tensor  # [B,F,N,2] or [B,M,F,N,2]
    key_padding_mask: Optional[torch.Tensor]  # [B,S]
    mode_logits: Optional[torch.Tensor] = None  # [B,N,M]


class VehicleTrajectoryTransformer(nn.Module):
    """
    Modular Transformer encoder for vehicle trajectory prediction.

    Stage-0 baseline (default config):
      - uses only `positions` [B, F, N, 2]
      - predicts per-token deltas [B, F, N, 2]

     configurable conditioning features:
      - `velocity` [B, F, N, 2]
      - `heading`  [B, F, N, 1] (rad) or [B, F, N, 2]
      - `agent_type` [B, N] int

    Map/traffic-light tokens are scaffolded as optional extra tokens appended
    to the sequence, but are not required for stage-0/1.
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
        map_polyline_encoder: str = "mean",
        map_polyline_max_points: int = 20,
        map_use_type_embedding: bool = False,
        map_type_vocab_size: int = 64,
        tl_use_state_embedding: bool = False,
        tl_state_vocab_size: int = 16,
        tl_use_time_embedding: bool = False,
        tl_time_vocab_size: int = 16,
        map_integration: str = "append",
        tl_integration: str = "append",
        use_agent_map_feat: bool = False,
        agent_map_feat_dim: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_time = int(max_time)
        self.max_agents = int(max_agents)
        self.obs_len = int(obs_len)
        self.num_modes = int(num_modes)
        self.use_velocity = bool(use_velocity)
        self.use_heading = bool(use_heading)
        self.use_type = bool(use_type)
        self.use_map = bool(use_map)
        self.use_traffic_lights = bool(use_traffic_lights)
        self.map_feat_dim = int(map_feat_dim)
        self.traffic_light_feat_dim = int(traffic_light_feat_dim)
        self.map_use_polyline_encoder = bool(map_use_polyline_encoder)
        self.map_polyline_encoder = str(map_polyline_encoder)
        self.map_polyline_max_points = int(map_polyline_max_points)
        self.map_use_type_embedding = bool(map_use_type_embedding)
        self.map_type_vocab_size = int(map_type_vocab_size)
        self.tl_use_state_embedding = bool(tl_use_state_embedding)
        self.tl_state_vocab_size = int(tl_state_vocab_size)
        self.tl_use_time_embedding = bool(tl_use_time_embedding)
        self.tl_time_vocab_size = int(tl_time_vocab_size)
        self.map_integration = str(map_integration)
        self.tl_integration = str(tl_integration)
        self.use_agent_map_feat = bool(use_agent_map_feat)
        self.agent_map_feat_dim = int(agent_map_feat_dim)
        valid_poly_enc = {"mean", "attn"}
        if self.map_polyline_encoder not in valid_poly_enc:
            raise ValueError(f"map_polyline_encoder must be one of {sorted(valid_poly_enc)}, got {self.map_polyline_encoder!r}")

        valid_integrations = {"append", "cross_attn", "cross_attn_per_layer"}
        if self.map_integration not in valid_integrations:
            raise ValueError(f"map_integration must be one of {sorted(valid_integrations)}, got {self.map_integration!r}")
        if self.tl_integration not in valid_integrations:
            raise ValueError(f"tl_integration must be one of {sorted(valid_integrations)}, got {self.tl_integration!r}")

        self.pos_proj = nn.Linear(2, self.d_model)
        self.vel_proj = nn.Linear(2, self.d_model) if self.use_velocity else None
        self.heading_proj = HeadingProjection(self.d_model) if self.use_heading else None
        self.agent_map_proj = nn.Linear(self.agent_map_feat_dim, self.d_model) if self.use_agent_map_feat else None
        self.type_emb = nn.Embedding(int(type_vocab_size), self.d_model) if self.use_type else None

        self.time_emb = nn.Embedding(self.max_time, self.d_model)
        self.agent_emb = nn.Embedding(self.max_agents, self.d_model)
        self.dropout = nn.Dropout(float(dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,  # [B, S, D]
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        self.head = nn.Linear(self.d_model, 2)
        self.mode_head = nn.Linear(self.d_model, self.num_modes) if self.num_modes > 1 else None
        self.mm_decoder = (
            MultiModeDecoder(d_model=self.d_model, num_modes=self.num_modes, dropout=float(dropout))
            if self.num_modes > 1
            else None
        )

        # keeps the agent token sequence length fixed (F*N)
        # injects map/tl context via a separate attention op
        self._map_ctx_attn: Optional[nn.MultiheadAttention] = None
        self._map_ctx_attn_layers: Optional[nn.ModuleList] = None
        self._map_ctx_ln_q: Optional[nn.LayerNorm] = None
        self._map_ctx_ln_kv: Optional[nn.LayerNorm] = None
        if self.use_map and self.map_integration in {"cross_attn", "cross_attn_per_layer"}:
            if self.map_integration == "cross_attn_per_layer":
                self._map_ctx_attn_layers = nn.ModuleList(
                    [
                        nn.MultiheadAttention(self.d_model, int(num_heads), dropout=float(dropout), batch_first=True)
                        for _ in range(int(num_layers))
                    ]
                )
            else:
                self._map_ctx_attn = nn.MultiheadAttention(
                    self.d_model, int(num_heads), dropout=float(dropout), batch_first=True
                )
            self._map_ctx_ln_q = nn.LayerNorm(self.d_model)
            self._map_ctx_ln_kv = nn.LayerNorm(self.d_model)

        self._tl_ctx_attn: Optional[nn.MultiheadAttention] = None
        self._tl_ctx_attn_layers: Optional[nn.ModuleList] = None
        self._tl_ctx_ln_q: Optional[nn.LayerNorm] = None
        self._tl_ctx_ln_kv: Optional[nn.LayerNorm] = None
        if self.use_traffic_lights and self.tl_integration in {"cross_attn", "cross_attn_per_layer"}:
            if self.tl_integration == "cross_attn_per_layer":
                self._tl_ctx_attn_layers = nn.ModuleList(
                    [
                        nn.MultiheadAttention(self.d_model, int(num_heads), dropout=float(dropout), batch_first=True)
                        for _ in range(int(num_layers))
                    ]
                )
            else:
                self._tl_ctx_attn = nn.MultiheadAttention(
                    self.d_model, int(num_heads), dropout=float(dropout), batch_first=True
                )
            self._tl_ctx_ln_q = nn.LayerNorm(self.d_model)
            self._tl_ctx_ln_kv = nn.LayerNorm(self.d_model)

        # Map / traffic-light token projections.
        #
        # Backward compatible by default:
        # - if map_use_type_embedding/tl_use_state_embedding are False,
        #   we project the raw token feature vector as a whole (previous behavior).
        # - if True, we treat the last channel as a categorical id and embed it.
        self._map_proj_raw: Optional[nn.Linear] = None
        self._map_xy_proj: Optional[nn.Linear] = None
        self._map_type_emb: Optional[nn.Embedding] = None
        self._map_poly_pt_mlp: Optional[nn.Sequential] = None
        self._map_poly_fuse: Optional[nn.Sequential] = None
        self._map_poly_pos_emb: Optional[nn.Embedding] = None
        self._map_poly_attn: Optional[nn.MultiheadAttention] = None
        self._map_poly_ln: Optional[nn.LayerNorm] = None
        if self.use_map:
            if self.map_use_polyline_encoder:
                # Encodes a polyline with P points into one token.
                # Input expected: roadgraph_static [B,G,P,3] where last channel is type_id.
                self._map_poly_pt_mlp = nn.Sequential(
                    nn.Linear(2, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                if self.map_polyline_encoder == "attn":
                    self._map_poly_pos_emb = nn.Embedding(self.map_polyline_max_points, self.d_model)
                    self._map_poly_attn = nn.MultiheadAttention(
                        self.d_model, int(num_heads), dropout=float(dropout), batch_first=True
                    )
                    self._map_poly_ln = nn.LayerNorm(self.d_model)
                # Fuse pooled polyline feature (+ optional type embedding) back to d_model.
                # If type embedding is off, we still accept a scalar type_id by concatenation.
                in_dim = self.d_model + (self.d_model if self.map_use_type_embedding else 1)
                self._map_poly_fuse = nn.Sequential(
                    nn.Linear(in_dim, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.d_model),
                )
                if self.map_use_type_embedding:
                    self._map_type_emb = nn.Embedding(self.map_type_vocab_size, self.d_model)
            else:
                if self.map_use_type_embedding:
                    self._map_xy_proj = nn.Linear(2, self.d_model)
                    self._map_type_emb = nn.Embedding(self.map_type_vocab_size, self.d_model)
                else:
                    self._map_proj_raw = nn.Linear(self.map_feat_dim, self.d_model)

        self._tl_proj_raw: Optional[nn.Linear] = None
        self._tl_xy_proj: Optional[nn.Linear] = None
        self._tl_state_emb: Optional[nn.Embedding] = None
        self._tl_time_emb: Optional[nn.Embedding] = None
        if self.use_traffic_lights:
            if self.tl_use_state_embedding:
                self._tl_xy_proj = nn.Linear(2, self.d_model)
                self._tl_state_emb = nn.Embedding(self.tl_state_vocab_size, self.d_model)
            else:
                self._tl_proj_raw = nn.Linear(self.traffic_light_feat_dim, self.d_model)
            if self.tl_use_time_embedding:
                self._tl_time_emb = nn.Embedding(self.tl_time_vocab_size, self.d_model)

    @staticmethod
    def _ids_from_last_channel(x: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """
        Interprets the last channel of x as a categorical id stored as float and returns [..] long ids.
        """
        ids = x.round().to(dtype=torch.long)
        return ids.clamp(min=0, max=max(int(vocab_size) - 1, 0))

    def _encode_map_tokens(self, map_tokens: torch.Tensor) -> torch.Tensor:
        if self.map_use_polyline_encoder:
            raise RuntimeError("map_use_polyline_encoder=true: expected roadgraph_static/padding_mask_static, not map_tokens")
        if not self.map_use_type_embedding:
            if self._map_proj_raw is None:
                raise RuntimeError("map raw proj missing despite use_map=true")
            if int(map_tokens.shape[-1]) != int(self.map_feat_dim):
                raise ValueError(
                    f"map_tokens last dim must be map_feat_dim={self.map_feat_dim}, got {int(map_tokens.shape[-1])}"
                )
            return self._map_proj_raw(map_tokens)

        # Expect at least [x,y,type_id] as the first 3 channels.
        if int(map_tokens.shape[-1]) < 3:
            raise ValueError(f"map_tokens last dim must be >=3 when using type embedding, got {int(map_tokens.shape[-1])}")
        if self._map_xy_proj is None or self._map_type_emb is None:
            raise RuntimeError("map embedding layers missing despite map_use_type_embedding=true")
        xy = map_tokens[..., 0:2]
        type_ids = self._ids_from_last_channel(map_tokens[..., 2], self.map_type_vocab_size)
        return self._map_xy_proj(xy) + self._map_type_emb(type_ids)

    def _encode_map_polylines(
        self,
        roadgraph_static: torch.Tensor,  # [B,G,P,3]
        padding_mask_static: torch.Tensor,  # [B,G,P] True=pad
    ) -> torch.Tensor:
        if not self.map_use_polyline_encoder:
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

        # Optional polyline self-attention with positional embeddings.
        if self.map_polyline_encoder == "attn":
            if self._map_poly_attn is None or self._map_poly_pos_emb is None or self._map_poly_ln is None:
                raise RuntimeError("map_polyline_encoder=attn but polyline attn layers are missing")
            B, G, P, D = pts.shape
            Pmax = min(int(self.map_polyline_max_points), P)
            pts = pts[:, :, :Pmax, :]
            pm = padding_mask_static[:, :, :Pmax]
            pos_ids = torch.arange(Pmax, device=pts.device, dtype=torch.long)
            pts = pts + self._map_poly_pos_emb(pos_ids)[None, None, :, :]
            x = pts.reshape(B * G, Pmax, D)
            kpm = _ensure_not_all_masked(pm.reshape(B * G, Pmax))
            attn_out, _ = self._map_poly_attn(
                self._map_poly_ln(x),
                self._map_poly_ln(x),
                self._map_poly_ln(x),
                key_padding_mask=kpm,
            )
            x = x + self.dropout(attn_out)
            x = x.reshape(B, G, Pmax, D)
            valid_local = (~pm).to(dtype=pts.dtype).unsqueeze(-1)
            denom = valid_local.sum(dim=2).clamp(min=1.0)
            pooled = (x * valid_local).sum(dim=2) / denom
        else:
            denom = valid.sum(dim=2).clamp(min=1.0)  # [B,G,1]
            pooled = pts.sum(dim=2) / denom  # [B,G,D]

        if self.map_use_type_embedding:
            if self._map_type_emb is None:
                raise RuntimeError("map_type_emb missing despite map_use_type_embedding=true")
            type_ids = self._ids_from_last_channel(roadgraph_static[:, :, 0, 2], self.map_type_vocab_size)  # [B,G]
            type_feat = self._map_type_emb(type_ids)  # [B,G,D]
            fused_in = torch.cat([pooled, type_feat], dim=-1)  # [B,G,2D]
        else:
            type_scalar = roadgraph_static[:, :, 0, 2:3]  # [B,G,1]
            fused_in = torch.cat([pooled, type_scalar], dim=-1)  # [B,G,D+1]

        return self._map_poly_fuse(fused_in)  # [B,G,D]

    def _encode_tl_tokens(self, tl_tokens: torch.Tensor) -> torch.Tensor:
        if not self.tl_use_state_embedding:
            if self._tl_proj_raw is None:
                raise RuntimeError("traffic-light raw proj missing despite use_traffic_lights=true")
            if int(tl_tokens.shape[-1]) < int(self.traffic_light_feat_dim):
                raise ValueError(
                    f"tl_tokens last dim must be >= traffic_light_feat_dim={self.traffic_light_feat_dim}, got {int(tl_tokens.shape[-1])}"
                )
            raw = tl_tokens[..., : self.traffic_light_feat_dim]
            return self._tl_proj_raw(raw)

        # Expect at least [x,y,state_id] as the first 3 channels.
        if int(tl_tokens.shape[-1]) < 3:
            raise ValueError(f"tl_tokens last dim must be >=3 when using state embedding, got {int(tl_tokens.shape[-1])}")
        if self._tl_xy_proj is None or self._tl_state_emb is None:
            raise RuntimeError("traffic-light embedding layers missing despite tl_use_state_embedding=true")
        xy = tl_tokens[..., 0:2]
        state_ids = self._ids_from_last_channel(tl_tokens[..., 2], self.tl_state_vocab_size)
        out = self._tl_xy_proj(xy) + self._tl_state_emb(state_ids)
        if self.tl_use_time_embedding:
            if self._tl_time_emb is None:
                raise RuntimeError("tl_time_emb missing despite tl_use_time_embedding=true")
            if int(tl_tokens.shape[-1]) < 4:
                raise ValueError("tl_use_time_embedding=true requires tl_tokens to have a 4th channel time_id")
            time_ids = self._ids_from_last_channel(tl_tokens[..., 3], self.tl_time_vocab_size)
            out = out + self._tl_time_emb(time_ids)
        return out

    def _ctx_attend(
        self,
        *,
        q_tokens: torch.Tensor,  # [B,S,D]
        kv_tokens: torch.Tensor,  # [B,K,D]
        kv_key_padding_mask: torch.Tensor,  # [B,K] True=pad
        attn: nn.MultiheadAttention,
        ln_q: nn.LayerNorm,
        ln_kv: nn.LayerNorm,
    ) -> torch.Tensor:
        q = ln_q(q_tokens)
        kv = ln_kv(kv_tokens)
        kpm = _ensure_not_all_masked(kv_key_padding_mask.to(dtype=torch.bool, device=q_tokens.device))
        ctx, _ = attn(q, kv, kv, key_padding_mask=kpm, need_weights=False)
        return q_tokens + self.dropout(ctx)

    def forward(
        self,
        *,
        positions: torch.Tensor,  # [B, F, N, 2]
        valid_mask: Optional[torch.Tensor] = None,  # [B, F, N]
        velocity: Optional[torch.Tensor] = None,  # [B, F, N, 2]
        heading: Optional[torch.Tensor] = None,  # [B, F, N, 1|2]
        agent_type: Optional[torch.Tensor] = None,  # [B, N]
        agent_map_feat: Optional[torch.Tensor] = None,  # [B, N, Dm]
        map_tokens: Optional[torch.Tensor] = None,  # [B, S_map, D_map]
        map_key_padding_mask: Optional[torch.Tensor] = None,  # [B, S_map] True=pad
        roadgraph_static: Optional[torch.Tensor] = None,  # [B, G, P, C]
        padding_mask_static: Optional[torch.Tensor] = None,  # [B, G, P] True=pad
        tl_tokens: Optional[torch.Tensor] = None,  # [B, S_tl, D_tl]
        tl_key_padding_mask: Optional[torch.Tensor] = None,  # [B, S_tl] True=pad
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if positions.dim() != 4 or positions.size(-1) != 2:
            raise ValueError(f"positions must be [B,F,N,2], got {tuple(positions.shape)}")

        B, F, N, _ = positions.shape
        if N > self.max_agents:
            raise ValueError(
                f"N={N} exceeds max_agents={self.max_agents}; set data.max_agents/model.max_agents consistently."
            )
        if F > self.max_time:
            raise ValueError(
                f"F={F} exceeds max_time={self.max_time}; set data.num_frames/model.max_time consistently."
            )

        token = self.pos_proj(positions)

        if self.use_velocity:
            if velocity is None:
                raise ValueError("use_velocity=true but velocity is None")
            token = token + self.vel_proj(velocity)

        if self.use_heading:
            if heading is None:
                raise ValueError("use_heading=true but heading is None")
            token = token + self.heading_proj(heading)

        frame_idx = torch.arange(F, device=positions.device)
        agent_idx = torch.arange(N, device=positions.device)
        token = token + self.time_emb(frame_idx)[None, :, None, :]
        token = token + self.agent_emb(agent_idx)[None, None, :, :]

        if self.use_type:
            if agent_type is None:
                raise ValueError("use_type=true but agent_type is None")
            if agent_type.shape != (B, N):
                raise ValueError(f"agent_type must be [B,N], got {tuple(agent_type.shape)}")
            type_tok = self.type_emb(agent_type.clamp(min=0))
            token = token + type_tok[:, None, :, :]

        if self.use_agent_map_feat:
            if agent_map_feat is None:
                raise ValueError("use_agent_map_feat=true but agent_map_feat is None")
            if agent_map_feat.shape[:2] != (B, N):
                raise ValueError(f"agent_map_feat must be [B,N,D], got {tuple(agent_map_feat.shape)}")
            if agent_map_feat.shape[-1] != self.agent_map_feat_dim:
                raise ValueError(f"agent_map_feat last dim must be {self.agent_map_feat_dim}, got {agent_map_feat.shape[-1]}")
            if self.agent_map_proj is None:
                raise RuntimeError("agent_map_proj missing despite use_agent_map_feat=true")
            token = token + self.agent_map_proj(agent_map_feat)[:, None, :, :]

        token = self.dropout(token)

        agent_tokens = token.view(B, F * N, self.d_model)
        key_padding_mask = None
        if valid_mask is not None:
            if valid_mask.shape != (B, F, N):
                raise ValueError(f"valid_mask must be [B,F,N], got {tuple(valid_mask.shape)}")
            key_padding_mask = (~valid_mask.to(dtype=torch.bool)).view(B, F * N)

        tokens = agent_tokens
        append_tokens = []
        append_masks = []
        per_layer_map: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (tok, kpm)
        per_layer_tl: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (tok, kpm)

        if self.use_map:
            map_tok: torch.Tensor
            map_mask: torch.Tensor
            if self.map_use_polyline_encoder:
                if roadgraph_static is None:
                    raise ValueError("use_map=true and map_use_polyline_encoder=true but roadgraph_static is None")
                if padding_mask_static is None:
                    raise ValueError("use_map=true and map_use_polyline_encoder=true but padding_mask_static is None")
                rg = roadgraph_static.to(dtype=torch.float32)
                pm = padding_mask_static.to(dtype=torch.bool)
                map_tok = self._encode_map_polylines(rg, pm)  # [B,G,D]
                map_mask = pm.all(dim=2)  # [B,G] True=pad
            else:
                if map_tokens is None:
                    raise ValueError("use_map=true but map_tokens is None")
                if map_key_padding_mask is None:
                    map_key_padding_mask = torch.zeros(
                        B, map_tokens.shape[1], dtype=torch.bool, device=map_tokens.device
                    )
                map_tok = self._encode_map_tokens(map_tokens)
                map_mask = map_key_padding_mask.to(dtype=torch.bool, device=map_tok.device)

            if self.map_integration == "append":
                append_tokens.append(map_tok)
                append_masks.append(map_mask)
            elif self.map_integration == "cross_attn":
                if self._map_ctx_attn is None or self._map_ctx_ln_q is None or self._map_ctx_ln_kv is None:
                    raise RuntimeError("map_integration=cross_attn but cross-attn layers are not initialized")
                tokens = self._ctx_attend(
                    q_tokens=tokens,
                    kv_tokens=map_tok,
                    kv_key_padding_mask=map_mask,
                    attn=self._map_ctx_attn,
                    ln_q=self._map_ctx_ln_q,
                    ln_kv=self._map_ctx_ln_kv,
                )
            else:
                per_layer_map = (map_tok, map_mask)

        if self.use_traffic_lights:
            if tl_tokens is None:
                raise ValueError("use_traffic_lights=true but tl_tokens is None")
            if tl_key_padding_mask is None:
                tl_key_padding_mask = torch.zeros(
                    B, tl_tokens.shape[1], dtype=torch.bool, device=tl_tokens.device
                )
            tl_tok = self._encode_tl_tokens(tl_tokens)
            tl_mask = tl_key_padding_mask.to(dtype=torch.bool, device=tl_tok.device)

            if self.tl_integration == "append":
                append_tokens.append(tl_tok)
                append_masks.append(tl_mask)
            elif self.tl_integration == "cross_attn":
                if self._tl_ctx_attn is None or self._tl_ctx_ln_q is None or self._tl_ctx_ln_kv is None:
                    raise RuntimeError("tl_integration=cross_attn but cross-attn layers are not initialized")
                tokens = self._ctx_attend(
                    q_tokens=tokens,
                    kv_tokens=tl_tok,
                    kv_key_padding_mask=tl_mask,
                    attn=self._tl_ctx_attn,
                    ln_q=self._tl_ctx_ln_q,
                    ln_kv=self._tl_ctx_ln_kv,
                )
            else:
                per_layer_tl = (tl_tok, tl_mask)

        if append_tokens:
            tokens = torch.cat([tokens] + append_tokens, dim=1)
            if key_padding_mask is not None:
                kpm = [key_padding_mask]
                for tok, m in zip(append_tokens, append_masks):
                    kpm.append(m.to(dtype=torch.bool, device=tokens.device))
                key_padding_mask = torch.cat(kpm, dim=1)

        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        use_per_layer_ctx = per_layer_map is not None or per_layer_tl is not None
        if not use_per_layer_ctx:
            feats = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        else:
            x = tokens
            for idx, layer in enumerate(self.encoder.layers):
                x = layer(x, src_key_padding_mask=key_padding_mask)
                if per_layer_map is not None:
                    if self._map_ctx_ln_q is None or self._map_ctx_ln_kv is None:
                        raise RuntimeError("map_integration=cross_attn_per_layer but cross-attn layers are not initialized")
                    if self._map_ctx_attn_layers is None:
                        raise RuntimeError("map_integration=cross_attn_per_layer but per-layer attn is missing")
                    map_tok, map_mask = per_layer_map
                    x = self._ctx_attend(
                        q_tokens=x,
                        kv_tokens=map_tok,
                        kv_key_padding_mask=map_mask,
                        attn=self._map_ctx_attn_layers[idx],
                        ln_q=self._map_ctx_ln_q,
                        ln_kv=self._map_ctx_ln_kv,
                    )
                if per_layer_tl is not None:
                    if self._tl_ctx_ln_q is None or self._tl_ctx_ln_kv is None:
                        raise RuntimeError("tl_integration=cross_attn_per_layer but cross-attn layers are not initialized")
                    if self._tl_ctx_attn_layers is None:
                        raise RuntimeError("tl_integration=cross_attn_per_layer but per-layer attn is missing")
                    tl_tok, tl_mask = per_layer_tl
                    x = self._ctx_attend(
                        q_tokens=x,
                        kv_tokens=tl_tok,
                        kv_key_padding_mask=tl_mask,
                        attn=self._tl_ctx_attn_layers[idx],
                        ln_q=self._tl_ctx_ln_q,
                        ln_kv=self._tl_ctx_ln_kv,
                    )
            if self.encoder.norm is not None:
                x = self.encoder.norm(x)
            feats = x
        agent_feats = feats[:, : F * N].view(B, F, N, self.d_model)

        if self.num_modes <= 1 or self.mm_decoder is None or self.mode_head is None:
            deltas = self.head(agent_feats).view(B, F, N, 2)
            return VehicleTrajectoryTransformerOutput(deltas=deltas, key_padding_mask=key_padding_mask)

        # Multimodal: treat agents as A and time as T
        x = agent_feats.permute(0, 2, 1, 3).contiguous()  # [B,N,F,D]
        hM = self.mm_decoder(x)  # [B,M,N,F,D]
        deltasM = self.head(hM).permute(0, 1, 3, 2, 4).contiguous()  # [B,M,F,N,2]

        # Mode logits per agent: use last observed timestep embedding (index obs_len-1 if provided via max_time)
        # Training script decides which timestep to use; we default to frame 0 if unknown.
        obs_idx = max(0, min(int(self.obs_len) - 1, F - 1))
        logits = self.mode_head(x[:, :, obs_idx, :])  # [B,N,M]
        return VehicleTrajectoryTransformerOutput(deltas=deltasM, key_padding_mask=key_padding_mask, mode_logits=logits)


__all__ = ["VehicleTrajectoryTransformer", "VehicleTrajectoryTransformerOutput"]
