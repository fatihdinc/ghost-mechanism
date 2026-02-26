#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:19:40 2025

@author: dinc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:42:29 2025

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.ndimage import minimum_filter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset



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
        #nn.init.xavier_uniform_(self.W_in.bias)

        # with torch.no_grad():
        #     self.W_rec.weight.fill_diagonal_(0)
    
    def forward(self, inp):
        # inp --> B, T, in_dim
        
        if self.normalize:
            with torch.no_grad():
                norm = self.W_out.weight.norm(p=1)/self.c
                self.W_out.weight.div_(norm)
                if self.W_out.bias is not None:
                    self.W_out.bias.div_(norm)
            
        
        time = inp.shape[1]
        firing_rates = F.tanh(torch.randn(inp.shape[0], self.hid_dim).to(inp.device) * self.firing_rate_sigma)
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
            pred_out = F.sigmoid(self.W_out(firing_rate@self.S_l)) # B, D_out
        else:
            pred_out = F.sigmoid(self.W_out(firing_rate)) # B, D_out


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
    
    def calculate_accuracy(self, pred, gt, reaction_start, reaction_end):
        # x / sum(x) for obtaining pred probability distribution.
        reaction_accuracy = 1 - F.l1_loss(pred[:, reaction_start:reaction_end], gt[:, reaction_start:reaction_end])

        return torch.round(reaction_accuracy, decimals=4).item()


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

def get_lpus(rnn, r,kappas):
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
    
    dk = np.zeros((2, len(kappas), len(kappas)))

    for i, k1 in enumerate(kappas):
        for j, k2 in enumerate(kappas):
            kappa = np.array([k1, k2])

            # recurrent drive
            drive = M @ kappa + b
            nonlin = N@np.tanh(drive)

            # dynamics
            dk[:, i, j] = (-kappa + nonlin)
            
    kappas_emp = np.zeros([4,r.shape[1],2])
    kappas_emp[0,:,:] = (r[0,:,:] @ N.T)
    kappas_emp[1,:,:] = (r[1,:,:] @ N.T)
    kappas_emp[2,:,:] = (r[2,:,:] @ N.T)
    kappas_emp[3,:,:] = (r[3,:,:] @ N.T)
    kappas_abs = np.sqrt(np.sum(dk**2,0)).T

    return kappas, dk,kappas_emp,kappas_abs,N.T

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss_and_grad(seed, results_dir="results_dmts", epochs=None, X=100, save=None):
    """
    Plot normalized loss and gradient history for a given seed on a log scale.

    Args:
        seed (int): which seed folder to load
        results_dir (str): root results folder
        epochs (list[int]): epoch indices to highlight with vertical dashed lines
        X (int): stride for plotting points (default=100)
        save (str): if not None, save figure to this path
    """
    data = np.load(os.path.join(results_dir, f"seed_{seed}", "final.npz"), allow_pickle=True)
    loss_history = np.array(data["loss_history"])
    grad_history = np.array(data["grad_history"])

    # Normalize
    loss_norm = loss_history / (np.max(loss_history) + 1e-12)
    grad_norm = grad_history / (np.max(grad_history) + 1e-12)

    # Downsample
    idx_ds = np.arange(len(loss_norm))[::X]
    loss_ds = loss_norm[::X]
    grad_ds = grad_norm[::X]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx_ds, loss_ds, label="Loss (normalized)", color=colors[0])
    ax.plot(idx_ds, grad_ds, label="Grad norm (normalized)", color=colors[1], alpha=0.7)

    if epochs is not None:
        for e in epochs:
            if 0 <= e < len(loss_norm):
                ax.axvline(e, color="gray", linestyle="--", alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized value (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Seed {seed} Training (stride={X})")
    ax.legend()

    if save is not None:
        plt.savefig(save)
        
 
    plt.show()
    
    

def F_kappa(k,M,N,b):
    drive = M @ k + b
    return -k + N @ np.tanh(drive)

def J_kappa(k,M,N,b):
    drive = M @ k + b
    sech2 = 1.0 - np.tanh(drive)**2     # derivative tanh
    D = np.diag(sech2)
    return -np.eye(2) + N @ D @ M


def newton_minimize_kappa(kappa0, M, N, b,
                          tol=1e-16,
                          max_iter=100,
                          damping=1e-6,
                          verbose=False):
    """
    Minimize kinetic energy E(kappa)=1/2 ||F(kappa)||^2
    using Gauss–Newton descent.

    Parameters
    ----------
    kappa0 : array shape (2,)
        Initial guess
    M, N, b : as defined in your reduction
    tol : float
        Gradient tolerance
    max_iter : int
        Maximum iterations
    damping : float
        Small Tikhonov regularization
    verbose : bool

    Returns
    -------
    kappa_star : ndarray (2,)
    info : dict
    """

    kappa = np.array(kappa0, dtype=float)

    for it in range(max_iter):

        F = F_kappa(kappa, M, N, b)
        J = J_kappa(kappa, M, N, b)

        grad = J.T @ F
        grad_norm = np.linalg.norm(grad)

        if verbose:
            print(f"iter {it}: |grad|={grad_norm:.3e}")

        if grad_norm < tol:
            break

        # ----- exact Hessian -----

        H = J.T @ J

        z = M @ kappa + b
        tanh_z = np.tanh(z)
        s = 1.0 - tanh_z**2
        t = -2.0 * tanh_z * s

        # second derivative contribution
        for i in range(2):              # dimension of F
            Hi = np.zeros((2,2))
            for k in range(len(z)):
                Mk = M[k, :].reshape(2,1)
                Hi += N[i, k] * t[k] * (Mk @ Mk.T)
            H += F[i] * Hi

        # small regularization for safety
        H += damping * np.eye(2)

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(H) @ grad

        kappa = kappa - delta

    info = {
        "iterations": it,
        "grad_norm": grad_norm,
        "F_norm": np.linalg.norm(F_kappa(kappa, M, N, b))
    }

    return kappa, info


seed = 0
hid_dim = 100
save = None
conf = None
step = 10
results_dir = 'results_dmts'
task = 'dms'
transpose = 0
input_interval=0.030
delay_interval=0.080
reaction_interval=0.050
delta_t=5e-3
alpha=0.5 
K=2
    
range_val = [-5,5]
epoch = 1600
    

"""
Plot flow maps and output traces from a saved model at a chosen epoch.

Args:
    seed (int): which seed folder to load
    epoch (int): checkpoint to load
    hid_dim (int): hidden dimension
    range_val (list[float]): [min,max] for x/y axis zoom
    save (str): if not None, save figure to this path
    conf (bool): whether to show confidence/decision regions
    step (int): stride for quiver arrows
    results_dir (str): root results folder
    task (str): dataset type
    input_interval, delay_interval, reaction_interval, delta_t: dataset params
    alpha (float): time constant ratio
    K (int): low-rank dimension
"""
# Recreate dataset
data = Single_Dataset(num_of_samples_per_each_class=10,
                      input_interval=input_interval,
                      delay_interval=delay_interval,
                      reaction_interval=reaction_interval,
                      post_reaction_interval=0.,
                      delta_t=delta_t,
                      task=task)
inp, gt_out = data.dataset[:, 0], data.dataset[:, 1]

# Reload model
rnn = RNN(in_dim=1, hid_dim=hid_dim, out_dim=1,
          alpha=alpha, noise_sigma=0, firing_rate_sigma=0.1,
          low_rank=K, normalize=False)
model_path = os.path.join(results_dir, f"seed_{seed}", "models", f"model_epoch{epoch}.pt")
rnn.load_state_dict(torch.load(model_path))

with torch.no_grad():
    pred_out, r = rnn(inp)

kappas = np.linspace(range_val[0], range_val[1], 400)

kappas, dk, kappas_emp, kappa_abs, Nt = get_lpus(
    rnn, r.detach().cpu().numpy(), kappas
)

X, Y = np.meshgrid(kappas, kappas, indexing="ij")
U, V = dk[0], dk[1]

fig, axs = plt.subplots(2, 1, figsize=(8, 10),
                        gridspec_kw={"height_ratios": [3, 1]})

# Flow field
Xs, Ys = X[::step, ::step], Y[::step, ::step]
Us, Vs = U[::step, ::step], V[::step, ::step]
mags = np.sqrt(Us**2 + Vs**2)
mags[mags == 0] = 1

axs[0].quiver(Xs, Ys, Us/mags, Vs/mags,
              angles="xy", scale=50,
              width=0.002, color="black")

im = axs[0].imshow(np.log10(kappa_abs),
                   origin="lower",
                   extent=[kappas[0], kappas[-1],
                           kappas[0], kappas[-1]],
                   cmap="viridis",
                   alpha=0.6, vmin=-1, vmax=1)

fig.colorbar(im, ax=axs[0], pad=0.1)


# Trajectories
print(kappas_emp.shape[0])
cl = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(kappas_emp.shape[0]):
    axs[0].scatter(kappas_emp[i,:,0], kappas_emp[i,:,1], s=15, c=cl[i % len(cl)])


# Detect local minima
footprint = np.ones((3, 3))
local_min = (kappa_abs == minimum_filter(kappa_abs,
                                         footprint=footprint))

minima_y, minima_x = np.where(local_min)

print(minima_x)

# Reduced matrices
with torch.no_grad():
    W = (rnn.S_l @ rnn.S_r).detach().cpu().numpy().T
    b = rnn.W_in.bias.detach().cpu().numpy()
    U_svd, S_svd, Vt_svd = np.linalg.svd(W)
    M = U_svd[:, :2] @ np.diag(S_svd[:2])
    N_mat = Vt_svd[:2, :]


h = 1e-5
eig_line_length = 0.02  # adjust visually if needed

for idx in range(len(minima_x)):

    k1 = kappas[minima_x[idx]]
    k2 = kappas[minima_y[idx]]
    kappa_star = np.array([k1, k2])

    # Numerical Jacobian
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = 1.0
        J[:, j] = (F_kappa(kappa_star + h * e,M,N_mat,b)
                   - F_kappa(kappa_star - h * e,M,N_mat,b)) / (2 * h)

    eigvals, eigvecs = np.linalg.eig(J)
    eigvals2, eigvecs = np.linalg.eig(J_kappa(kappa_star,M,N_mat,b))
    #print(kappa_star,eigvals,eigvals2)
    
    kappa_new,_ = newton_minimize_kappa(kappa_star,M,N_mat,b)
    eigvals2, eigvecs = np.linalg.eig(J_kappa(kappa_new,M,N_mat,b))
    print("kappa:", kappa_new, "Speed:",
          np.linalg.norm(F_kappa(kappa_new,M,N_mat,b)),
          "\n \t eigenvalues (before and after)",
          eigvals,eigvals2)
    # Normalize eigenvectors
    for i in range(2):
        eigvecs[:, i] /= np.linalg.norm(eigvecs[:, i])

    # Identify slow eigenvector
    slow_idx = np.argmin(np.abs(np.real(eigvals)))

    for i in range(2):

        vec = eigvecs[:, i]
        color = "red" if i == slow_idx else "black"

        x_vals = [k1 - eig_line_length * vec[0],
                  k1 + eig_line_length * vec[0]]
        y_vals = [k2 - eig_line_length * vec[1],
                  k2 + eig_line_length * vec[1]]

        axs[0].plot(x_vals, y_vals,
                    color=color,
                    linewidth=2)

    axs[0].scatter(k1, k2,
                   marker="x",
                   color="purple",
                   s=60)

axs[0].set_title(f"Flow map with eigenvectors, seed {seed}, epoch {epoch}")

# Output traces
axs[1].plot(gt_out[0, :, 0].detach().numpy(),
            color="blue", ls="-")
axs[1].plot(pred_out[0, :, 0].detach().numpy(),
            color="blue", ls="--")
axs[1].plot(gt_out[1, :, 0].detach().numpy(),
            color="red", ls="-")
axs[1].plot(pred_out[1, :, 0].detach().numpy(),
            color="red", ls="--")

if range_val is not None:
    axs[0].set_xlim(range_val)
    axs[0].set_ylim(range_val)

plt.tight_layout()
if save is not None:
    plt.savefig(save)

plt.show()



