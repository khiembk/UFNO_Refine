from ufno import Net3d
import torch 
import torch.nn as nn 

from models.Seimic_UFNO import SeismicUFNO_encode


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


class Coupled_Model(nn.Module):
     
    def __init__(self, mode1, mode2, mode3, width, pretrained_model:nn.Module = None):
        super().__init__()

        self.ufno = pretrained_model
        freeze(self.ufno)
        self.ufno.eval()
        self.main_model= SeismicUFNO_encode(mode1, mode2, mode3, width)
        
        self.feature_proj = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=1)
        )
        
        # fuse layer
        self.fuse = nn.Sequential(
            nn.Conv3d(2 * width, width, kernel_size=1),
            nn.GELU()
        )
        

    def forward(self, x):
        
        with torch.no_grad():
            t_feat = self.ufno.encode(x)
        
        t_feat = self.feature_proj(t_feat)

        s_feat = self.main_model.encode(x)
        # print("sfeat:", s_feat.shape)
        # print("t_feat:",t_feat.shape)
        assert s_feat.shape == t_feat.shape
        feat = torch.cat([s_feat, t_feat], dim=1)
        feat = self.fuse(feat)
        out = self.main_model.decode(feat)
        return out