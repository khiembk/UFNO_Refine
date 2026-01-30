from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SeismicTextDataset(Dataset):
    def __init__(self, a, u, texts, roberta_name="distilroberta-base", max_len=64):
        assert len(texts) == a.shape[0] == u.shape[0]
        self.a = a
        self.u = u

        tok = AutoTokenizer.from_pretrained(roberta_name)
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.input_ids = enc["input_ids"]
        self.attn_mask = enc["attention_mask"]

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx], self.u[idx], self.input_ids[idx], self.attn_mask[idx]
