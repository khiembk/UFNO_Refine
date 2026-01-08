from ufno import Net3d
import torch 
import torch.nn as nn 
import Seimic_UFNO

class Coupled_Model(nn.Module):
     
    def __init__(self, mode1, mode2, mode3, width):
        super().__init__()

        self.ufno = Net3d(mode1, mode2, mode3, width)
        self.main_model= Seimic_UFNO(mode1, mode2, mode3, width)
        # Project latent field → receivers
        self.proj = nn.Sequential(
            nn.Conv3d(width, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(64, 24, kernel_size=1)  # output time channels
        )

        # Receiver grid projection
        self.pool = nn.AdaptiveAvgPool3d((24, 151, 101))

    def forward(self, x):
        """
        x: [B, 96, 200, 24, 12]
        """
        x = self.ufno.encode(x)   # [B, 96, 200, 24, width]
        x = x.permute(0, 4, 3, 1, 2)  # [B, width, 24, 96, 200]

        x = self.proj(x)          # [B, 24, 24, 96, 200]
        x = self.pool(x)          # [B, 24, 24, 151, 101]

        # collapse time-feature dim
        x = x.mean(dim=2)         # [B, 24, 151, 101]

        return x