import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer


class RobertaTeacher(nn.Module):
    """
    Frozen RoBERTa teacher for semantic guidance
    """

    def __init__(self, model_name="roberta-base", out_dim=36):
        super().__init__()

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.llm = RobertaModel.from_pretrained(model_name)

        # Freeze RoBERTa
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        # Adapter (trainable)
        self.adapter = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )

    @torch.no_grad()
    def encode_text(self, texts, device):
        """
        texts: list[str]
        return: [B, 768]
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        outputs = self.llm(**tokens)
        cls_emb = outputs.last_hidden_state[:, 0]  # [B, 768]
        return cls_emb

    def forward(self, texts, device):
        """
        texts: list[str]
        return: [B, out_dim]
        """
        with torch.no_grad():
            cls_emb = self.encode_text(texts, device)

        return self.adapter(cls_emb)

class CoupledWaveModel(nn.Module):
    def __init__(self, teacher, student, sg_dim, llm_feat_dim):
        super().__init__()

        self.teacher = teacher  # RobertaTeacher

        self.fusion = nn.Sequential(
            nn.Linear(sg_dim + llm_feat_dim, 64),
            nn.GELU()
        )

        self.student = student
    def forward(self, sg_feat, texts, device):
        """
        sg_feat: [B, sg_dim]
        texts:   list[str] (length B)
        """

        teacher_feat = self.teacher(texts, device)  # [B, llm_feat_dim]

        x = torch.cat([sg_feat, teacher_feat], dim=1)
        x = self.fusion(x)
        return self.student(x)
