import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

from datetime import datetime
from scipy.ndimage import minimum_filter
import os

class RNN(nn.Module):
    def __init__(self,
                 in_dim=1,
                 hid_dim=100,
                 out_dim=1,
                 alpha:float=0.1,
                 noise_sigma:float=0.001,
                 firing_rate_sigma:float=0.1,
                 low_rank=None,
                 c = 1,
                 normalize = False):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.normalize = normalize
        self.c = c

        self.noise_sigma = noise_sigma

        self.alpha = alpha
        self.firing_rate_sigma = firing_rate_sigma
        self.low_rank = low_rank

        self.W_in = nn.Linear(in_dim, hid_dim, bias=True)

        # self.W_rec = nn.Linear(hid_dim, hid_dim, bias=True)

        if low_rank:
            self.S_l = nn.Parameter(torch.randn(hid_dim, low_rank))
            self.S_r = nn.Parameter(torch.randn(low_rank, hid_dim))

            nn.init.normal_(self.S_l, mean=0.0, std=1.0/np.sqrt(hid_dim))
            nn.init.normal_(self.S_r, mean=0.0, std=1.0)
        else:
            self.W_rec = nn.Parameter(torch.randn(hid_dim, hid_dim))
            nn.init.xavier_uniform_(self.W_rec)

        
        if low_rank:
            self.W_out = nn.Linear(low_rank, out_dim, bias=True)
            
        else:
 
            self.W_out = nn.Linear(hid_dim, out_dim, bias=True)
        
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.normal_(self.W_in.bias, mean=0.0, std=0.1)
    
    def forward(self, inp):
        # inp --> B, T, in_dim
        
        if self.normalize:
            with torch.no_grad():
                norm = self.W_out.weight.norm(p=1)/self.c
                self.W_out.weight.div_(norm)
                if self.W_out.bias is not None:
                    self.W_out.bias.div_(norm)
            
        
        time = inp.shape[1]
        firing_rates = F.tanh( torch.randn(inp.shape[0], self.hid_dim).to(inp.device) * self.firing_rate_sigma)
        pred_out_list = []
        firing_rate_list = []

        for t in range(time):
            pred, firing_rates = self.forward_step(inp[:, t], firing_rates)

            pred_out_list.append(pred)
            firing_rate_list.append(firing_rates)
        
        return torch.stack(pred_out_list).permute(1, 0, 2), torch.stack(firing_rate_list).permute(1, 0, 2) # B, T, out_dim - B, T, hid_dim
    
    def forward_step(self, inp, firing_rate):
        # input --> B(Batch), D_in(R^{N_{in}})
        # fire_rate --> B(Batch), N+N_2

        # o^(t) = sigmoid(W_out @ r(t))
 
    
        if self.low_rank: 
            pred_out = F.tanh(self.W_out(firing_rate@self.S_l)) # B, D_out
        else:
            pred_out = F.tanh(self.W_out(firing_rate)) # B, D_out


        # r[t + Δt] = (1 − α) * r[t] + α * tanh(W_rec @ r[t] + W_in @ I_in[t] + ε)
        hid = self.W_in(inp)

        if self.low_rank:
            self.W_rec_lr = torch.matmul(self.S_l, self.S_r)
            rec = torch.matmul(firing_rate, self.W_rec_lr)
        else:
            rec = torch.matmul(firing_rate, self.W_rec)


        noise = torch.randn(rec.shape).to(inp.device) * self.noise_sigma
        z = F.tanh(hid+rec+noise)

        pred_firing_rates = (1 - self.alpha) * firing_rate + self.alpha * z
        return pred_out, pred_firing_rates


class Single_Dataset(Dataset):
    # Delayed Cue Matching Task
    def __init__(self, 
                 num_of_samples_per_each_class=1,
                 input_interval=0.5, 
                 delay_interval=1.0, 
                 reaction_interval=1.0, 
                 post_reaction_interval=0.5,
                 delta_t=5e-3,
                 task='dcdt'):
        super().__init__()

        self.batch_size = num_of_samples_per_each_class

        self.task = task

        self.input_interval = input_interval
        self.delay_interval = delay_interval
        self.reaction_interval = reaction_interval
        self.post_reaction_interval = post_reaction_interval

        self.delta_t = delta_t

        self.dataset = self.create_dataset() # B, [in, out], T, 2

    def ground_truths(self):
        return self.create_dataset(1)
    
    def get_num_of_samples(self):
        return {
            "num_in_samples": int(self.input_interval / self.delta_t),
            "num_delay_samples": int(self.delay_interval / self.delta_t),
            "num_reaction_samples": int(self.reaction_interval / self.delta_t),
            "num_post_reaction_samples": int(self.post_reaction_interval / self.delta_t)
        }

    def create_dataset(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        num_of_samples = self.get_num_of_samples()

        if self.task == 'dcdt':
            input_1 = torch.cat((torch.ones(num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1), torch.zeros(num_of_samples['num_reaction_samples'], 1)), dim=0)
            input_2 = torch.cat((torch.ones(num_of_samples['num_in_samples'], 1)*-1, torch.zeros(num_of_samples['num_delay_samples'], 1), torch.zeros(num_of_samples['num_reaction_samples'], 1)), dim=0)


            out_1 = torch.cat((torch.zeros(num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1),  torch.ones(num_of_samples['num_reaction_samples'], 1)), dim=0)
            out_2 = torch.cat((torch.zeros(num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1),  -1*torch.ones(num_of_samples['num_reaction_samples'], 1)), dim=0)
            inputs = torch.cat((input_1.unsqueeze(0), input_2.unsqueeze(0)), dim=0).unsqueeze(1)
            outs = torch.cat((out_1.unsqueeze(0), out_2.unsqueeze(0)), dim=0).unsqueeze(1)
        else:
            input_1 = torch.cat((torch.ones(num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1), torch.ones(num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_reaction_samples'], 1)), dim=0)
            input_2 = torch.cat((torch.ones(num_of_samples['num_in_samples'], 1)*-1, torch.zeros(num_of_samples['num_delay_samples'], 1), torch.ones(num_of_samples['num_in_samples'], 1)*-1, torch.zeros(num_of_samples['num_reaction_samples'], 1)), dim=0)
            input_3 = torch.cat((torch.ones(num_of_samples['num_in_samples'], 1)*-1, torch.zeros(num_of_samples['num_delay_samples'], 1), torch.ones(num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_reaction_samples'], 1)), dim=0)
            input_4 = torch.cat((torch.ones(num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1), torch.ones(num_of_samples['num_in_samples'], 1)*-1, torch.zeros(num_of_samples['num_reaction_samples'], 1)), dim=0)

            out_1 = torch.cat((torch.zeros(2*num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1),  torch.ones(num_of_samples['num_reaction_samples'], 1)), dim=0)
            out_2 = torch.cat((torch.zeros(2*num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1),  torch.ones(num_of_samples['num_reaction_samples'], 1)), dim=0)
            out_3 = torch.cat((torch.zeros(2*num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1),  torch.zeros(num_of_samples['num_reaction_samples'], 1)), dim=0)
            out_4 = torch.cat((torch.zeros(2*num_of_samples['num_in_samples'], 1), torch.zeros(num_of_samples['num_delay_samples'], 1),  torch.zeros(num_of_samples['num_reaction_samples'], 1)), dim=0)

            inputs = torch.cat((input_1.unsqueeze(0), input_2.unsqueeze(0), input_3.unsqueeze(0), input_4.unsqueeze(0)), dim=0).unsqueeze(1)
            outs = torch.cat((out_1.unsqueeze(0), out_2.unsqueeze(0), out_3.unsqueeze(0), out_4.unsqueeze(0)), dim=0).unsqueeze(1)
        return torch.cat((inputs, outs), dim=1).repeat(batch_size, 1, 1, 1) # B*2, 2, T, 1

    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1] # input, out

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def binarize(arr, lower_thresh=-0.33, upper_thresh=0.33):
    """
    Binarize a NumPy array to -1, 0, or 1.
    Values <= lower_thresh → -1
    Values between thresholds → 0
    Values >= upper_thresh → 1
    """
    out = np.zeros_like(arr)
    out[arr >= upper_thresh] = 1
    out[arr <= lower_thresh] = -1
    return out



def get_lpus(rnn, r):
    """
    Compute dynamics of latent processing units (LPUs) for a 2D reduction.

    Args:
        rnn: trained RNN model with attributes S_l, S_r, W_in
        tau: time constant
        N:   scaling factor
        f:   nonlinearity (default: tanh)
        u:   input vector u(t) (default: zeros)

    Returns:
        kappas: 1D array of values used for grid
        dk:     array of shape [2, len(kappas), len(kappas)]
                representing RHS of dynamics dκ/dt
    """
    with torch.no_grad():
        # Effective recurrent weights
        W = (rnn.S_l @ rnn.S_r).detach().cpu().numpy().T
        b = rnn.W_in.bias.detach().cpu().numpy()

        # 2D reduction via SVD
        U, S, Vt = np.linalg.svd(W)
        M = U[:, :2] @ np.diag(S[:2])   # recurrent reduced
        N = Vt[:2, :]             # projection matrix
        


    # κ grid
    kappas = np.linspace(-7.5, 7.5, 400)
    dk = np.zeros((2, len(kappas), len(kappas)))

    for i, k1 in enumerate(kappas):
        for j, k2 in enumerate(kappas):
            kappa = np.array([k1, k2])

            # recurrent drive
            drive = M @ kappa + b
            nonlin = N@np.tanh(drive)

            # dynamics
            dk[:, i, j] = (-kappa + nonlin)
            
    kappas_emp = np.zeros([2,r.shape[1],2])
    kappas_emp[0,:,:] = (r[0,:,:] @ N.T) # One type of trial
    kappas_emp[1,:,:] = (r[-1,:,:] @ N.T) # Another type for trial
    kappas_abs = (np.sum(dk**2,0).T) # Transpose needed to match the imshow dimensions later, which are transposed

    return kappas, dk,kappas_emp,kappas_abs,N.T

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Main code 

# Hyperparameters 


N = 100
K=2
num_epochs = 100000
ops = 'SGD'
device = 'cpu'

input_interval = 0.010
delay_interval = 0.050
reaction_interval = 0.050

delta_t = 5e-3
time_decay = 10e-3
alpha = delta_t / time_decay # 0.5


learning_rate=1e-3
k = 0



os.makedirs(f"results_dcdt/seed_{k}/figure", exist_ok=True)
os.makedirs(f"results_dcdt/seed_{k}/models", exist_ok=True)

seed_everything(seed=k)

rnn = RNN(in_dim=1,
            hid_dim=N,
            out_dim=1,
            alpha=alpha,
            noise_sigma=0,
            firing_rate_sigma=0.3,
            low_rank=K,
            normalize = False)

data = Single_Dataset(num_of_samples_per_each_class=500,
                        input_interval=input_interval, 
                        delay_interval=delay_interval, 
                        reaction_interval=reaction_interval, 
                        post_reaction_interval=0.,
                        delta_t=delta_t,
                        task='dcdt')

if ops == 'Adam':
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
else:
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

loss_history = []
grad_history = []
accuracy_history = []

inp, gt_out = data.dataset[:, 0], data.dataset[:, 1] # data.dataset.shape = B, (inp, out), T, 1
inp = inp.to(device)
gt_out = gt_out.to(device)
rnn.to(device)

pbar = tqdm(range(num_epochs))

for epoch in pbar:
    
    optimizer.zero_grad()

    pred_out, _ = rnn(inp)
    loss = criterion(pred_out, gt_out)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    grad_val = torch.norm(rnn.S_l.grad).cpu().item()+torch.norm(rnn.S_r.grad).cpu().item()
    grad_history.append(grad_val)
    accuracy_history.append( np.mean( binarize(pred_out.cpu().detach().numpy()) == gt_out.cpu().detach().numpy() ) )

    pbar.set_postfix(loss=f"{loss.item():.6f}", grad=f"{grad_val:.6f}")
    
    if epoch % 1000 == 0:
        plt.figure(figsize = (20,8))
        plt.subplot(231)
        plt.semilogy((loss_history),color=colors[0])
        plt.semilogy((grad_history),color = colors[1],alpha = 0.1)
        plt.subplot(234)
        plt.plot(gt_out[0,:,0].cpu().detach().numpy(),color = 'blue',ls='-')
        plt.plot(gt_out[1,:,0].cpu().detach().numpy(),color = 'red',ls='-')
        plt.plot(pred_out[0,:,0].cpu().detach().numpy(),color = 'blue',ls='--')
        plt.plot(pred_out[1,:,0].cpu().detach().numpy(),color = 'red',ls='--')
        
        plt.subplot(132)
        pred_out, r = rnn(inp)
        kappas, dk,kappas_emp,kappa_abs,Nt = get_lpus(rnn,r.detach().cpu().numpy())
        
        
    
        X, Y = np.meshgrid(kappas, kappas,indexing="ij")
        U = dk[0]
        V = dk[1]
        step=5
        Xs = X[::step, ::step]
        Ys = Y[::step, ::step]
        Us = U[::step, ::step]
        Vs = V[::step, ::step]
        
        
        # Normalize arrows
        mags = np.sqrt(Us**2 + Vs**2)
        mags[mags == 0] = 1.0   # avoid divide by zero
        Us_norm = Us / mags
        Vs_norm = Vs / mags
        
        plt.quiver(
            Xs, Ys,
            Us_norm, Vs_norm,
            angles="xy",
            scale=50,       # adjust to control arrow size
            width=0.001,
            color="black"
        )
        im = plt.imshow(
            np.log(kappa_abs),                       # the 2D array (magnitude field)
            origin="lower",                  # so (0,0) is bottom-left
            extent=[kappas[0], kappas[-1],   # x-axis span
                    kappas[0], kappas[-1]],  # y-axis span
            cmap="viridis", 
            alpha=0.6,interpolation = 'none',vmin = -5,vmax = 5
        )
        plt.colorbar(im, ax=plt.gca(), orientation="horizontal", pad=0.1)
        plt.scatter(kappas_emp[0,:,0],kappas_emp[0,:,1],1)
        plt.scatter(kappas_emp[1,:,0],kappas_emp[1,:,1],1)
        with torch.no_grad():
            W_out = rnn.W_out.weight.detach().cpu().numpy() @ rnn.S_l.T.detach().cpu().numpy()
            b_out = rnn.W_out.bias.detach().cpu().numpy()
        
        # Define grid in kappa-space
        X, Y = np.meshgrid(kappas, kappas, indexing="ij")
        Z = np.zeros_like(X)
        
        # Compute o = W_out r + b_out, with r ≈ N^T kappa
        for i in range(len(kappas)):
            for j in range(len(kappas)):
                kappa = np.array([X[i,j], Y[i,j]])
                r_proj = Nt @ kappa       # back-project kappa into full r-space
                Z[i,j] = (W_out @ r_proj + b_out).squeeze()
        
        plt.contourf(X, Y, Z, levels=[-.55, 0, .55], colors=["red","green"], alpha=0.3)
        
        footprint = np.ones((3,3))
        local_min = (kappa_abs == minimum_filter(kappa_abs, footprint=footprint))
        
        # get indices of minima
        minima_y, minima_x = np.where(local_min)
        # convert indices -> actual kappa values
        x_coords = kappas[minima_x]
        y_coords = kappas[minima_y]
        
        plt.scatter(x_coords, y_coords, marker="x",
                    color = 'purple', s=20)
        
        
        plt.subplot(133)
        
        x_min = np.min(kappas_emp[:,:,0])
        x_max = np.max(kappas_emp[:,:,0])
        y_min = np.min(kappas_emp[:,:,1])
        y_max = np.max(kappas_emp[:,:,1])
        
        step=5
        Xs = X[::step, ::step]
        Ys = Y[::step, ::step]
        Us = U[::step, ::step]
        Vs = V[::step, ::step]
        
        
        # Normalize arrows
        mags = np.sqrt(Us**2 + Vs**2)
        mags[mags == 0] = 1.0   # avoid divide by zero
        Us_norm = Us / mags
        Vs_norm = Vs / mags
        
        plt.quiver(
            Xs, Ys,
            Us_norm, Vs_norm,
            angles="xy",
            scale=20,       # adjust to control arrow size
            width=0.003,
            color="black"
        )
        im = plt.imshow(
            np.log(kappa_abs),                       # the 2D array (magnitude field)
            origin="lower",                  # so (0,0) is bottom-left
            extent=[kappas[0], kappas[-1],   # x-axis span
                    kappas[0], kappas[-1]],  # y-axis span
            cmap="viridis", 
            alpha=0.6,interpolation='none',vmin = -5,vmax = 5
        )
        
        plt.xlim([x_min-2,x_max+2])
        plt.ylim([y_min-2,y_max+2])
        plt.scatter(kappas_emp[0,:,0],kappas_emp[0,:,1],4)
        plt.scatter(kappas_emp[1,:,0],kappas_emp[1,:,1],4)
        local_min = (kappa_abs == minimum_filter(kappa_abs, footprint=footprint))
        
        # get indices of minima
        minima_y, minima_x = np.where(local_min)
        # convert indices -> actual kappa values
        x_coords = kappas[minima_x]
        y_coords = kappas[minima_y]
        
        plt.scatter(x_coords, y_coords, marker="x",
                    color = 'purple', s=20)
        plt.contourf(X, Y, Z, levels=[-.55, 0, .55], colors=["red","green"], alpha=0.3)
        
        
    
        
        

        #plt.xlim([-3,3])
        #plt.ylim([-5,-2])
        plt.savefig(f"results_dcdt/seed_{k}/figure/plot_epoch{epoch}.pdf")        
        plt.show()
        
    if epoch % 1 == 0: 
        torch.save(rnn.state_dict(), f"results_dcdt/seed_{k}/models/model_epoch{epoch}.pt")


np.savez(f'results_dcdt/seed_{k}/final.npz',loss_history=loss_history,grad_history=grad_history,accuracy_history=accuracy_history)


torch.save(rnn.state_dict(), f"results_dcdt/seed_{k}/model_final.pt")
plt.figure(figsize = (20,8))
plt.subplot(231)
plt.semilogy((loss_history),color=colors[0])
plt.semilogy((grad_history),color = colors[1],alpha = 0.1)
plt.subplot(234)
plt.plot(gt_out[0,:,0].detach().numpy(),color = 'blue',ls='-')
plt.plot(gt_out[1,:,0].detach().numpy(),color = 'red',ls='-')
plt.plot(pred_out[0,:,0].detach().numpy(),color = 'blue',ls='--')
plt.plot(pred_out[1,:,0].detach().numpy(),color = 'red',ls='--')

plt.subplot(132)
pred_out, r = rnn(inp)
kappas, dk,kappas_emp,kappa_abs,Nt = get_lpus(rnn,r.detach().cpu().numpy())



X, Y = np.meshgrid(kappas, kappas,indexing="ij")
U = dk[0]
V = dk[1]
step=5
Xs = X[::step, ::step]
Ys = Y[::step, ::step]
Us = U[::step, ::step]
Vs = V[::step, ::step]


# Normalize arrows
mags = np.sqrt(Us**2 + Vs**2)
mags[mags == 0] = 1.0   # avoid divide by zero
Us_norm = Us / mags
Vs_norm = Vs / mags

plt.quiver(
    Xs, Ys,
    Us_norm, Vs_norm,
    angles="xy",
    scale=50,       # adjust to control arrow size
    width=0.001,
    color="black"
)
im = plt.imshow(
    np.log(kappa_abs),                       # the 2D array (magnitude field)
    origin="lower",                  # so (0,0) is bottom-left
    extent=[kappas[0], kappas[-1],   # x-axis span
            kappas[0], kappas[-1]],  # y-axis span
    cmap="viridis", 
    alpha=0.6,interpolation = 'none',vmin = -5,vmax = 5
)
plt.colorbar(im, ax=plt.gca(), orientation="horizontal", pad=0.1)
plt.scatter(kappas_emp[0,:,0],kappas_emp[0,:,1],1)
plt.scatter(kappas_emp[1,:,0],kappas_emp[1,:,1],1)
with torch.no_grad():
    W_out = rnn.W_out.weight.detach().cpu().numpy() @ rnn.S_l.T.detach().cpu().numpy()
    b_out = rnn.W_out.bias.detach().cpu().numpy()

# Define grid in kappa-space
X, Y = np.meshgrid(kappas, kappas, indexing="ij")
Z = np.zeros_like(X)

# Compute o = W_out r + b_out, with r ≈ N^T kappa
for i in range(len(kappas)):
    for j in range(len(kappas)):
        kappa = np.array([X[i,j], Y[i,j]])
        r_proj = Nt @ kappa       # back-project kappa into full r-space
        Z[i,j] = (W_out @ r_proj + b_out).squeeze()

plt.contourf(X, Y, Z, levels=[-.55, 0, .55], colors=["red","green"], alpha=0.3)

footprint = np.ones((3,3))
local_min = (kappa_abs == minimum_filter(kappa_abs, footprint=footprint))

# get indices of minima
minima_y, minima_x = np.where(local_min)
# convert indices -> actual kappa values
x_coords = kappas[minima_x]
y_coords = kappas[minima_y]

plt.scatter(x_coords, y_coords, marker="x",
            color = 'purple', s=20)


plt.subplot(133)

x_min = np.min(kappas_emp[:,:,0])
x_max = np.max(kappas_emp[:,:,0])
y_min = np.min(kappas_emp[:,:,1])
y_max = np.max(kappas_emp[:,:,1])

step=5
Xs = X[::step, ::step]
Ys = Y[::step, ::step]
Us = U[::step, ::step]
Vs = V[::step, ::step]


# Normalize arrows
mags = np.sqrt(Us**2 + Vs**2)
mags[mags == 0] = 1.0   # avoid divide by zero
Us_norm = Us / mags
Vs_norm = Vs / mags

plt.quiver(
    Xs, Ys,
    Us_norm, Vs_norm,
    angles="xy",
    scale=20,       # adjust to control arrow size
    width=0.003,
    color="black"
)
im = plt.imshow(
    np.log(kappa_abs),                       # the 2D array (magnitude field)
    origin="lower",                  # so (0,0) is bottom-left
    extent=[kappas[0], kappas[-1],   # x-axis span
            kappas[0], kappas[-1]],  # y-axis span
    cmap="viridis", 
    alpha=0.6,interpolation='none',vmin = -5,vmax = 5
)

plt.xlim([x_min-2,x_max+2])
plt.ylim([y_min-2,y_max+2])
plt.scatter(kappas_emp[0,:,0],kappas_emp[0,:,1],4)
plt.scatter(kappas_emp[1,:,0],kappas_emp[1,:,1],4)
local_min = (kappa_abs == minimum_filter(kappa_abs, footprint=footprint))

# get indices of minima
minima_y, minima_x = np.where(local_min)
# convert indices -> actual kappa values
x_coords = kappas[minima_x]
y_coords = kappas[minima_y]

plt.scatter(x_coords, y_coords, marker="x",
            color = 'purple', s=20)
plt.contourf(X, Y, Z, levels=[-.55, 0, .55], colors=["red","green"], alpha=0.3)

plt.savefig(f"results_dcdt/seed_{k}/ffinal.pdf")        
plt.show()

