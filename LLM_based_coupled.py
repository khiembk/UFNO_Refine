import torch 
from utils.metric import NormalizedMRELoss, masked_r2,r2_score
from utils.data_loader import SeismicTextDataset
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from models.LLM_Bridge import Coupled_Model, RobertaGasTeacher

def load_gas_data():
    DATA_DIR = 'datasets'
    a = torch.load(f'{DATA_DIR}/sg_test_a.pt')
    u = torch.load(f'{DATA_DIR}/sg_test_u.pt')
    
    ntrain = int(0.8 * a.shape[0])
    train_a, val_a = a[:ntrain], a[ntrain:]
    train_u, val_u = u[:ntrain], u[ntrain:]
    print("Train:", train_a.shape, train_u.shape)
    print("Val  :", val_a.shape, val_u.shape)
    return train_a, train_u, val_a, val_u


def load_gen_data():
    DATA_DIR = 'datasets'
    a = torch.load(f'{DATA_DIR}/sg_test_a.pt')
    u = torch.load(f'{DATA_DIR}/seismic_test_u.pt')

    ntrain = int(0.8 * a.shape[0])
    train_a, val_a = a[:ntrain], a[ntrain:]
    train_u, val_u = u[:ntrain], u[ntrain:]

    print("Train:", train_a.shape, train_u.shape)
    print("Val  :", val_a.shape, val_u.shape)
    return train_a, train_u, val_a, val_u

@torch.no_grad()
def evaluate_gas_model(model, loader, device):
    model.eval()
    mre_total = 0.0
    r2_total = 0.0
    n_samples = 0
    lploss = NormalizedMRELoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        B = x.size(0)

        pred = model(x).view(-1, 96, 200, 24)
        mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)

        # sum MRE over samples in this batch
        mre_batch = 0.0
        for i in range(B):
            mre_batch += lploss(
                pred[i][mask[i]].reshape(1, -1),
                y[i][mask[i]].reshape(1, -1)
            )

        r2 = masked_r2(pred, y, mask)

        mre_total += float(mre_batch)
        r2_total += float(r2) * B
        n_samples += B

    return mre_total / n_samples, r2_total / n_samples

def save_roberta_teacher(model, path="checkpoints/roberta_gas_teacher.pt"):
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), path)

def load_roberta_teacher(cin, width=36, path="checkpoints/roberta_gas_teacher.pt", device="cuda"):
    model = RobertaGasTeacher(cin=cin, width=width).to(device)
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    return model

def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False
    m.eval()

def train_gas_roberta_teacher(epochs = 100):
    print("Load gas data...")
    train_a, train_u, val_a, val_u = load_gas_data()

    device = torch.device("cuda")
    cin = train_a.shape[-1]
    width = 36

    print("Create RoBERTa gas teacher...")
    teacher = RobertaGasTeacher(cin=cin, width=width, model_name="distilroberta-base").to(device)

    # prepare grid_dx exactly like you do
    print("Prepare grid_dx...")
    grid_x = train_a[0, 0, :, 0, -3]  # keep your indexing
    grid_dx = grid_x[1:-1] + grid_x[:-2] / 2 + grid_x[2:] / 2
    grid_dx = grid_dx[None, None, :, None].to(device)  # [1,1,198,1]

    
    lr = 1e-3
    scheduler_step = 2
    scheduler_gamma = 0.9

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_a, val_u),
        batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = NormalizedMRELoss()

    print("Begin gas pretraining (RoBERTa teacher)...")
    for ep in range(1, epochs + 1):
        teacher.train()
        total_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)

            optimizer.zero_grad()

            # mask like your code: [B,96,200,24]
            mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)

            # predict: [B,96,200,24]
            pred = teacher(x)

            # derivative terms along dim=2 (your code)
            dy = (y[:, :, 2:, :] - y[:, :, :-2, :]) / grid_dx
            dy_pred = (pred[:, :, 2:, :] - pred[:, :, :-2, :]) / grid_dx

            mask_dy = mask[:, :, :198, :]

            ori_loss = 0.0
            der_loss = 0.0

            # per-sample masked loss (same spirit as your implementation)
            for i in range(B):
                ori_loss = ori_loss + myloss(
                    pred[i][mask[i]].reshape(1, -1),
                    y[i][mask[i]].reshape(1, -1),
                )
                der_loss = der_loss + myloss(
                    dy_pred[i][mask_dy[i]].reshape(1, -1),
                    dy[i][mask_dy[i]].reshape(1, -1),
                )

            # IMPORTANT: normalize by batch size so scale is stable
            loss = (ori_loss + 0.5 * der_loss) / B

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * B
            n_samples += B

        scheduler.step()

        # quick validation with your evaluate function shape assumptions
        teacher.eval()
        with torch.no_grad():
            # reuse your evaluate logic but adapted: teacher(x) already [B,96,200,24]
            v_mre, v_r2 = evaluate_gas_teacher_roberta(teacher, val_loader, device)
        print(f"epoch {ep:03d} | gas teacher train {total_loss/n_samples:.6f} | val mre {v_mre:.6f} | val r2 {v_r2:.6f}")

    print("Save RoBERTa gas teacher...")
    save_roberta_teacher(teacher)
    return teacher
@torch.no_grad()
def evaluate_coupled_model(model, loader, device, batch_size):
    model.eval()
    mre_total = 0.0
    r2_total = 0.0
    n_batches = 0
    lploss = NormalizedMRELoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)

        mre = lploss(pred.reshape(pred.shape[0], -1),
            y.reshape(y.shape[0], -1))

        r2 = r2_score(pred.reshape(pred.shape[0], -1),
            y.reshape(y.shape[0], -1))

        mre_total += mre.item()
        r2_total += r2.item()
        n_batches += batch_size

    return mre_total / n_batches, r2_total / n_batches

@torch.no_grad()
def evaluate_gas_teacher_roberta(teacher, loader, device):
    teacher.eval()
    lploss = NormalizedMRELoss()

    mre_total = 0.0
    r2_total = 0.0
    n_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        B = x.size(0)

        pred = teacher(x)  # [B,96,200,24]
        mask = (x[:, :, :, 0:1, 0] != 0).repeat(1, 1, 1, 24)

        mre_batch = 0.0
        for i in range(B):
            mre_batch += lploss(
                pred[i][mask[i]].reshape(1, -1),
                y[i][mask[i]].reshape(1, -1),
            )

        r2 = masked_r2(pred, y, mask)

        mre_total += float(mre_batch)
        r2_total += float(r2) * B
        n_samples += B

    return mre_total / n_samples, r2_total / n_samples


def main():
    # 1) pretrain RoBERTa teacher on gas data
    gas_teacher = train_gas_roberta_teacher(epochs=1)

    # 2) freeze + train coupled wave model on wave/seismic data
    device = torch.device("cuda")
    mode1 = mode2 = mode3 = 10
    width = 36

    coupled_model = Coupled_Model(mode1, mode2, mode3, width, gas_teacher).to(device)

    train_a, train_u, val_a, val_u = load_gen_data()

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_a, val_u),
        batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(coupled_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    myloss = NormalizedMRELoss()

    for ep in range(1, 101):
        coupled_model.train()
        total, n = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)

            optimizer.zero_grad()
            pred = coupled_model(x)
            loss = myloss(pred.reshape(B, -1), y.reshape(B, -1))
            loss.backward()
            optimizer.step()

            total += float(loss) * B
            n += B

        scheduler.step()

        val_mre, val_r2 = evaluate_coupled_model(coupled_model, val_loader, device, batch_size)  # you can also fix this to use real B
        print(f"epoch {ep:03d} | wave train {total/n:.6f} | val mre {val_mre:.6f} | val r2 {val_r2:.6f}")


if __name__ =="__main__":
    main()
