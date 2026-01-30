import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from models.Seimic_UFNO import SeismicUFNO, SeismicUFNO_encode 

class RobertaGasTeacher(nn.Module):
    """
    Teacher pretrained on gas ODE:
      - encode(x) -> [B, width, D, H, T] feature map (like UFNO.encode)
      - forward(x) -> gas prediction [B, D, H, T]
    """
    def __init__(
        self,
        cin: int,
        width: int = 36,
        model_name: str = "distilroberta-base",
        patch: tuple = (12, 20, 4),   # makes tokens length <= 512 for (96,200,24)
    ):
        super().__init__()
        self.width = width
        self.cin = cin
        self.patch = patch

        # RoBERTa backbone (trainable during gas pretraining)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.d_model = self.backbone.config.hidden_size  # 768

        # numeric tensor -> token embeddings (Conv3D patches)
        # expects x as [B, Cin, D, H, T]
        self.patch_embed = nn.Conv3d(cin, self.d_model, kernel_size=patch, stride=patch, bias=False)

        # token outputs -> width channels
        self.to_width = nn.Linear(self.d_model, width)

        # refine in 3D feature space (same spirit as your feature_proj)
        self.feature_proj = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=1),
        )

        # gas regression head
        self.head = nn.Conv3d(width, 1, kernel_size=1)

    def encode(self, x):
        """
        x: [B, D, H, T, Cin]  (matches your current data layout)
        return: [B, width, D, H, T]
        """
        assert x.dim() == 5, f"Expected 5D x, got {x.shape}"
        B, D, H, T, Cin = x.shape
        assert Cin == self.cin, f"Cin mismatch: x has {Cin}, teacher expects {self.cin}"

        # -> [B, Cin, D, H, T] for Conv3d
        x3 = x.permute(0, 4, 1, 2, 3).contiguous()

        # patch embedding: [B, d_model, Dp, Hp, Tp]
        tok = self.patch_embed(x3)
        B, dm, Dp, Hp, Tp = tok.shape
        L = Dp * Hp * Tp
        if L > self.backbone.config.max_position_embeddings:
            raise ValueError(
                f"Token length {L} exceeds max_position_embeddings {self.backbone.config.max_position_embeddings}. "
                f"Use larger patch/stride to reduce tokens."
            )

        # -> [B, L, d_model]
        seq = tok.flatten(2).transpose(1, 2).contiguous()
        attn = torch.ones(B, L, device=x.device, dtype=torch.long)

        # transformer
        out = self.backbone(inputs_embeds=seq, attention_mask=attn).last_hidden_state  # [B, L, d_model]

        # -> width, reshape back to grid
        out = self.to_width(out)  # [B, L, width]
        feat_small = out.transpose(1, 2).reshape(B, self.width, Dp, Hp, Tp)

        # upsample to full resolution matching student features
        feat = F.interpolate(feat_small, size=(D, H, T), mode="trilinear", align_corners=False)
        feat = self.feature_proj(feat)
        return feat

    def forward(self, x):
        feat = self.encode(x)                     # [B, width, D, H, T]
        y = self.head(feat).squeeze(1)            # [B, D, H, T]
        return y



class Coupled_Model(nn.Module):
    def __init__(self, mode1, mode2, mode3, width, pretrained_teacher: nn.Module):
        super().__init__()
        pretrained_teacher.eval()
        self.teacher = pretrained_teacher
           # freeze after gas pretraining

        self.main_model = SeismicUFNO_encode(mode1, mode2, mode3, width)

        self.feature_proj = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=1)
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(2 * width, width, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        with torch.no_grad():
            t_feat = self.teacher.encode(x)   # [B,width,D,H,T]

        t_feat = self.feature_proj(t_feat)
        s_feat = self.main_model.encode(x)

        assert s_feat.shape == t_feat.shape, (s_feat.shape, t_feat.shape)

        feat = torch.cat([s_feat, t_feat], dim=1)
        feat = self.fuse(feat)

        out = self.main_model.decode(feat)
        return out
