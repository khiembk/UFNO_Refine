import torch
import torch.nn as nn
from lploss import LpLoss
import numpy as np
from torch.utils.data import random_split

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

class NormalizedMRELoss(torch.nn.Module):
    def __init__(self, p=2, eps=1e-8):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, pred, target):
        num = torch.norm(pred - target, p=self.p)
        den = torch.norm(target, p=self.p) + self.eps
        return num / den




def train_seismic_pinn(
    sg, target,
    device="cuda",
    epochs=100,
    batch_size=4
):
    dataset = SeismicPINNDataset(sg, target)

    # 80 / 20 split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    model = SeismicPINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = NormalizedMRELoss(p=2)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)   # [B, 96, 200, 24, 12]
            y = y.to(device)   # [B, 24, 151, 101]

            pred = model(x)

            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch:04d} | "
            f"Train NMRE {train_loss:.4e} | "
            f"Val NMRE {val_loss:.4e}"
        )

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
