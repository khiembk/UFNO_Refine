import torch
import torch.nn as nn
from ufno import Net3d


class SeismicUFNO(nn.Module):
    def __init__(self, mode1, mode2, mode3, width):
        super().__init__()

        # UFNO backbone (unchanged)
        self.ufno = Net3d(mode1, mode2, mode3, width)

        # Project spatial grid → receivers
        self.receiver_proj = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # Learnable receiver aggregation
        self.pool_x = nn.AdaptiveAvgPool1d(101)  # receivers

    def forward(self, x):
        """
        x: [B, 96, 200, 24, 12]
        return: [B, 24, 151, 101]
        """

        # UFNO forward
        # → [B, 96, 200, 24]
        field = self.ufno(x)

        B, Z, X, T = field.shape

        # Move time to front
        field = field.permute(0, 3, 1, 2)  # [B, 24, 96, 200]

        seismic = []

        for t in range(T):
            ft = field[:, t:t+1, :, :]      # [B, 1, 96, 200]
            ft = self.receiver_proj(ft)     # [B, 1, 96, 200]

            # Collapse depth (z)
            ft = ft.mean(dim=2)              # [B, 1, 200]

            # Map x → receivers
            ft = self.pool_x(ft)             # [B, 1, 101]

            seismic.append(ft)

        # Stack time
        seismic = torch.stack(seismic, dim=1)  # [B, 24, 1, 101]
        seismic = seismic.squeeze(2)            # [B, 24, 101]

        # Add time interpolation
        seismic = seismic.unsqueeze(2)           # [B, 24, 1, 101]
        seismic = nn.functional.interpolate(
            seismic,
            size=(151, 101),
            mode="bilinear",
            align_corners=False
        ).squeeze(2)

        return seismic
