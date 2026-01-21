import torch
import numpy as np
from ufno import *
from lploss import *
import os
torch.manual_seed(0)
np.random.seed(0)
from models.Seimic_UFNO import SeismicUFNO
from utils.metric import r2_score, mean_relative_error, evaluate_metrics, masked_mre, masked_r2,NormalizedMRELoss, count_parameters
import time

def load_data():
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
def evaluate(model, loader, device, batch_size):
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


def save_model(model):
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
    model.state_dict(),
    f"{save_dir}/gas_wave_last_model.pt")

def main():
    print("Load data...")
    train_a, train_u, val_a, val_u = load_data()

    print("create model...")
    mode1 = 10
    mode2 = 10
    mode3 = 10
    width = 36
    device = torch.device('cuda')
    model = SeismicUFNO(mode1, mode2, mode3, width)
    model.to(device)
    # prepare for calculating x direction derivatives 
    print("prepapre x...") 
    time_grid = np.cumsum(np.power(1.421245, range(24)))
    time_grid /= np.max(time_grid)
    grid_x = train_a[0,0,:,0,-3]
    grid_dx = grid_x[1:-1] + grid_x[:-2]/2 + grid_x[2:]/2
    grid_dx = grid_dx[None, None, :, None].to(device)
    print("setup params...")
    epochs = 100
    e_start = 0
    learning_rate = 0.001
    scheduler_step = 2
    scheduler_gamma = 0.9

    batch_size = 4
    print("load_train loader and val loader...")
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_a, val_u),
    batch_size=batch_size,
    shuffle=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = NormalizedMRELoss()

    train_l2 = 0.0
    print("Begin training...")
    for ep in range(1,epochs+1):
        model.train()
        train_l2 = 0
        counter = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
        
            optimizer.zero_grad()
            
            pred = model(x)
            
            loss = myloss(
            pred.reshape(pred.shape[0], -1),
            y.reshape(y.shape[0], -1)
        )
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

            counter += 1
            # if counter % 100 == 0:
            #     print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}')
        
        scheduler.step()
    
        
        val_mre, val_r2 = evaluate(model, val_loader, device, batch_size)
        print(f'epoch: {ep}, train loss: {train_l2/train_a.shape[0]:.4f}, val mre:{val_mre:.4f}, val r2:{val_r2:.4f}')
        lr_ = optimizer.param_groups[0]['lr']
    
    print("save model...")
    save_model(model)   

def measure_inference_time():
    device = "cuda"
    print("measure processing time")
    print("load data...")
    DATA_DIR = 'datasets'
    a = torch.load(f'{DATA_DIR}/sg_test_a.pt')
    u = torch.load(f'{DATA_DIR}/seismic_test_u.pt')
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a, u), batch_size= 4, shuffle=True)
    print("init model...")
    mode1 = 10
    mode2 = 10
    mode3 = 10
    width = 36
    device = torch.device('cuda')
    model = SeismicUFNO(mode1, mode2, mode3, width).to(device)
    model.eval()
    print("UFNO params: ", count_parameters(model))
    model.eval()
   
    
    x, y = next(iter(train_loader))
    x = x.to(device)

    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    torch.cuda.synchronize()

    # -------------------------
    # Measure inference time
    # -------------------------
    start = time.time()

    with torch.no_grad():
        _ = model(x)

    torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start

    print(f"Inference time (1 batch, B={x.shape[0]}): {elapsed*1000:.3f} ms")
    print(f"Per-sample time: {elapsed*1000/x.shape[0]:.3f} ms")

if __name__ == "__main__":
    measure_inference_time()
    #main()