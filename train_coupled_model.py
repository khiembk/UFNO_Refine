import torch
import numpy as np
from ufno import *
from lploss import *
import numpy as np 
torch.manual_seed(0)
np.random.seed(0)
from ufno import Net3d, Net3d_encode
from lploss import LpLoss
import os
from utils.metric import r2_score, mean_relative_error, evaluate_metrics, masked_mre, masked_r2
from models.BridgeModel import Coupled_Model


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

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


@torch.no_grad()
def evaluate_gas_model(model, loader, device, batch_size):
    model.eval()
    mre_total = 0.0
    r2_total = 0.0
    n_batches = 0
    lploss = LpLoss(size_average=False)
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x).view(-1, 96, 200, 24)

        mask = (x[:,:,:,0:1,0] != 0).repeat(1,1,1,24)
        
        for i in range(batch_size):
            mre =+ lploss(pred[i,...][mask[i,...]].reshape(1, -1), y[i,...][mask[i,...]].reshape(1, -1))


        r2 = masked_r2(pred, y, mask)

        mre_total += mre.item()
        r2_total += r2.item()
        n_batches += 1

    return mre_total / n_batches, r2_total / n_batches
@torch.no_grad()
def evaluate_coupled_model(model, loader, device, batch_size):
    model.eval()
    mre_total = 0.0
    r2_total = 0.0
    n_batches = 0
    lploss = LpLoss(size_average=False)
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)

        mre = lploss(pred.reshape(pred.shape[0], -1),
            y.reshape(y.shape[0], -1))

        r2 = r2_score(pred.reshape(pred.shape[0], -1),
            y.reshape(y.shape[0], -1))

        mre_total += mre.item()
        r2_total += r2.item()
        n_batches += 1

    return mre_total / n_batches, r2_total / n_batches

def save_gas_model(model):
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
    model.state_dict(),
    f"{save_dir}/gas_saturation_last_model.pt")

def train_gas_model():
    print("Load data...")
    train_a, train_u, val_a, val_u = load_gas_data()

    print("create model...")
    mode1 = 10
    mode2 = 10
    mode3 = 10
    width = 36
    device = torch.device('cuda')
    model = Net3d_encode(mode1, mode2, mode3, width)
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
    myloss = LpLoss(size_average=False)

    train_l2 = 0.0
    for ep in range(1,epochs+1):
        model.train()
        train_l2 = 0
        counter = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            dy = (y[:,:,2:,:] - y[:,:,:-2,:])/grid_dx
        
            optimizer.zero_grad()
        
            mask = (x[:,:,:,0:1,0]!=0).repeat(1,1,1,24)
            dy = (y[:,:,2:,:] - y[:,:,:-2,:])/grid_dx
            pred = model(x).view(-1,96,200,24)
            dy_pred = (pred[:,:,2:,:] - pred[:,:,:-2,:])/grid_dx
            ori_loss = 0
            der_loss = 0
        
        # original loss
            for i in range(batch_size):
                ori_loss += myloss(pred[i,...][mask[i,...]].reshape(1, -1), y[i,...][mask[i,...]].reshape(1, -1))

        # 1st derivative loss
            dy_pred = (pred[:,:,2:,:] - pred[:,:,:-2,:])/grid_dx
            mask_dy = mask[:,:,:198,:]
            for i in range(batch_size):
                der_loss += myloss(dy_pred[i,...][mask_dy[i,...]].reshape(1, -1), dy[i,...][mask_dy[i,...]].view(1, -1))

            loss = ori_loss + 0.5 * der_loss
        
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

            counter += 1
            # if counter % 100 == 0:
            #     print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}')
        
        scheduler.step()
    
        
        val_mre, val_r2 = evaluate_gas_model(model, val_loader, device, batch_size)
        print(f'epoch: {ep}, gas train loss: {train_l2/train_a.shape[0]:.4f}, gas val mre:{val_mre:.4f}, gas val r2:{val_r2:.4f}')
        lr_ = optimizer.param_groups[0]['lr']
    
    print("save model...")
    save_gas_model(model)
    return model   

def load_gas_model_pre_trained():
    mode1 = 10
    mode2 = 10
    mode3 = 10
    width = 36
    device = torch.device('cuda')
    model = Net3d_encode(mode1, mode2, mode3, width)
    
    state_dict = torch.load(
    "checkpoints/gas_saturation_last_model.pt",
    map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def main():
    #gas_model = train_gas_model()
    gas_model = load_gas_model_pre_trained()
    #gas_model.to(device)
    mode1 = 10
    mode2 = 10
    mode3 = 10
    width = 36
    print("create couple model")
    device = torch.device('cuda')
    coupled_model  =  Coupled_Model(mode1, mode2, mode3, width, gas_model)
    coupled_model.to(device)
    train_a, train_u, val_a, val_u = load_gen_data()
    print("prepapre x...") 
    time_grid = np.cumsum(np.power(1.421245, range(24)))
    time_grid /= np.max(time_grid)
    grid_x = train_a[0,0,:,0,-3]
    grid_dx = grid_x[1:-1] + grid_x[:-2]/2 + grid_x[2:]/2
    grid_dx = grid_dx[None, None, :, None].to(device)
    print("setup params...")
    epochs = 100
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
    optimizer = torch.optim.Adam(coupled_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=False)

    train_l2 = 0.0
    print("Begin training...")
    for ep in range(1,epochs+1):
        coupled_model.train()
        train_l2 = 0
        counter = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
        
            optimizer.zero_grad()
            
            pred = coupled_model(x)
            
            loss = myloss(
            pred.reshape(pred.shape[0], -1),
            y.reshape(y.shape[0], -1)
        )
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

            counter += 1
            
        
        scheduler.step()
    
        
        val_mre, val_r2 = evaluate_coupled_model(coupled_model, val_loader, device, batch_size)
        print(f'epoch: {ep}, train wave loss: {train_l2/train_a.shape[0]:.4f}, val wave mre:{val_mre:.4f}, val wave r2:{val_r2:.4f}')
        lr_ = optimizer.param_groups[0]['lr']
    
    



if __name__ == "__main__":
    main()