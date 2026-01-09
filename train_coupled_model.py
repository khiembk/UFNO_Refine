import torch
import numpy as np
from ufno import *
from lploss import *
import numpy as np 
torch.manual_seed(0)
np.random.seed(0)
from ufno import Net3d, Net3d_encode
from lploss import LpLoss

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def load_data():
    DATA_DIR = 'datasets'
    train_a = torch.load(f'{DATA_DIR}/dP_test_a_lite.pt')
    train_u = torch.load(f'{DATA_DIR}/dP_test_u_lite.pt')
    print(train_a.shape)
    print(train_u.shape)

    return train_a, train_u


def train_teacher_model():
    train_a, train_u = load_data()
    mode1 = 10
    mode2 = 10
    mode3 = 10
    width = 36
    device = torch.device('cuda')
    model = Net3d_encode(mode1, mode2, mode3, width)
    model.to(device)
    
    time_grid = np.cumsum(np.power(1.421245, range(24)))
    time_grid /= np.max(time_grid)
    grid_x = train_a[0,0,:,0,-3]
    grid_dx = grid_x[1:-1] + grid_x[:-2]/2 + grid_x[2:]/2
    grid_dx = grid_dx[None, None, :, None].to(device)

    epochs = 140
    e_start = 0
    scheduler_step = 4
    scheduler_gamma = 0.85
    learning_rate = 0.001
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
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
            current_batch_size = x.shape[0]
            for i in range(current_batch_size):
               ori_loss += myloss(pred[i][mask[i]].reshape(1, -1), y[i][mask[i]].reshape(1, -1))
               #ori_loss += myloss(pred[i,...][mask[i,...]].reshape(1, -1), y[i,...][mask[i,...]].reshape(1, -1))

        # 1st derivative loss
            dy_pred = (pred[:,:,2:,:] - pred[:,:,:-2,:])/grid_dx
            mask_dy = mask[:,:,:198,:]
            
            for i in range(current_batch_size):
                der_loss += myloss(dy_pred[i,...][mask_dy[i,...]].reshape(1, -1), dy[i,...][mask_dy[i,...]].view(1, -1))

            loss = ori_loss + 0.5 * der_loss
        
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

            counter += 1
            if counter % 100 == 0:
               print(f'epoch: {ep}, batch: {counter}/{len(train_loader)}, train loss: {loss.item()/batch_size:.4f}')
        
        scheduler.step()

        print(f'epoch: {ep}, train loss: {train_l2/train_a.shape[0]:.4f}')
    
        lr_ = optimizer.param_groups[0]['lr']