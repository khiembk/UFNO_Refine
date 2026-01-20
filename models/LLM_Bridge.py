import torch
import torch.nn as nn


class LLMTeacher(nn.Module):
    """
    LLM-based teacher that provides semantic guidance
    """

    def __init__(self, llm_dim=768, out_dim=36):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(llm_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

        # Teacher is frozen
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, llm_embedding):
        """
        llm_embedding: [B, llm_dim]
        return:        [B, out_dim]
        """
        return self.adapter(llm_embedding)


class WaveStudent(nn.Module):
    def __init__(self, in_dim, width=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.net(x)

class CoupledWaveModel(nn.Module):
    def __init__(self, teacher, sg_dim, llm_feat_dim):
        super().__init__()

        self.teacher = teacher

        # fuse sg + teacher features
        self.fusion = nn.Sequential(
            nn.Linear(sg_dim + llm_feat_dim, 64),
            nn.GELU()
        )

        self.student = WaveStudent(64)

    def forward(self, sg_feat, llm_embedding):
        """
        sg_feat:        [B, sg_dim]
        llm_embedding:  [B, llm_dim]
        """

        with torch.no_grad():
            teacher_feat = self.teacher(llm_embedding)

        x = torch.cat([sg_feat, teacher_feat], dim=1)
        x = self.fusion(x)
        return self.student(x)
