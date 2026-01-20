import torch 
from utils.metric import NormalizedMRELoss
from models.LLM_Bridge import LLMTeacher, CoupledWaveModel
from transformers import AutoTokenizer, AutoModel

class SeismicDataset(torch.utils.data.Dataset):
    def __init__(self, sg, seismic, llm_embeddings):
        """
        sg:            [N, 96, 200, 24, 12]
        seismic:       [N, 24, 151, 101]
        llm_embeddings:[N, 768]
        """
        self.sg = sg
        self.seismic = seismic
        self.llm_embeddings = llm_embeddings

    def __len__(self):
        return self.sg.shape[0]

    def __getitem__(self, idx):
        # compress SG
        sg_feat = self.sg[idx].mean(dim=(0,1,2))  # [12]

        # compress seismic (receiver energy)
        y = self.seismic[idx].abs().mean().unsqueeze(0)

        llm_embed = self.llm_embeddings[idx]

        return sg_feat, llm_embed, y

def train_wave_with_llm_teacher(
    sg, seismic, llm_embeddings,
    device="cuda",
    epochs=100,
    batch_size=8
):
    dataset = SeismicDataset(sg, seismic, llm_embeddings)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    teacher = LLMTeacher(llm_dim=llm_embeddings.shape[1], out_dim=36)
    model = CoupledWaveModel(
        teacher=teacher,
        sg_dim=12,
        llm_feat_dim=36
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = NormalizedMRELoss()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0

        for sg_feat, llm_embed, y in loader:
            sg_feat = sg_feat.to(device)
            llm_embed = llm_embed.to(device)
            y = y.to(device)

            pred = model(sg_feat, llm_embed)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch {ep:04d} | Loss {total / len(loader):.4e}")

    return model


