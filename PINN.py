import torch
import torch.nn as nn
from lploss import LpLoss
from utils.metric import r2_score
import numpy as np

def compress_sg(sg):
    # sg: [96, 200, 24, 12]
    # output: [24, 12]
    return sg.mean(dim=(0, 1))

class SeismicPINNDataset(torch.utils.data.Dataset):
    def __init__(self, sg, target, num_points=4000):
        """
        sg:     [N, 96, 200, 24, 12]
        target: [N, 24, 151, 101]
        """
        self.sg = sg
        self.target = target
        self.num_points = num_points

        self.N = sg.shape[0]

        self.t = torch.linspace(0, 1, 24)
        self.zr = torch.linspace(0, 1, 151)
        self.xr = torch.linspace(0, 1, 101)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sg = self.sg[idx]          # [96,200,24,12]
        target = self.target[idx]  # [24,151,101]

        # ---- compress sg field ----
        sg_feat = sg.mean(dim=(0,1))  # [24,12]

        # ---- sample receiver points ----
        it = torch.randint(0, 24, (self.num_points,))
        iz = torch.randint(0, 151, (self.num_points,))
        ix = torch.randint(0, 101, (self.num_points,))

        t = self.t[it]
        zr = self.zr[iz]
        xr = self.xr[ix]

        sg_t = sg_feat[it]  # [num_points, 12]
        y = target[it, iz, ix].unsqueeze(1)

        # input: (t, zr, xr, sg features)
        x = torch.cat(
            [t.unsqueeze(1), zr.unsqueeze(1), xr.unsqueeze(1), sg_t],
            dim=1
        )

        return x, y



class SeismicPINN(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_seismic_pinn(sg, target, device="cuda"):
    dataset = SeismicPINNDataset(sg, target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = SeismicPINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(2000):
        total = 0.0
        for x, y in loader:
            x = x.squeeze(0).to(device)
            y = y.squeeze(0).to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss {total/len(loader):.4e}")

    return model



def train():
    print("load data...")
    DATA_DIR = 'datasets'
    a = torch.load(f'{DATA_DIR}/sg_test_a.pt')
    u = torch.load(f'{DATA_DIR}/seismic_test_u.pt')
    print("train model...")
    model = train_seismic_pinn(
    a,
    u,
    device="cuda"
    )

if  __name__ == "__main__":
    train()
