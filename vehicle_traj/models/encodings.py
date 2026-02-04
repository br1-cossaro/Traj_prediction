from __future__ import annotations

import torch
import torch.nn as nn


class HeadingProjection(nn.Module):
    """
    Projects heading into model dimension.

    Supported inputs:
      - heading_rad: [..., 1] radians -> converted to [sin, cos]
      - heading_vec: [..., 2] already in sin/cos or any 2D representation
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2, d_model)

    def forward(self, heading: torch.Tensor) -> torch.Tensor:
        if heading.size(-1) == 1:
            h = heading
            heading_vec = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)
        elif heading.size(-1) == 2:
            heading_vec = heading
        else:
            raise ValueError(
                f"heading must have last dim 1 (rad) or 2 (vec), got {tuple(heading.shape)}"
            )
        return self.proj(heading_vec)


__all__ = ["HeadingProjection"]

