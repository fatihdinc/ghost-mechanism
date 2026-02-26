import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

from datetime import datetime


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
                 task='dcdt'):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.noise_sigma = noise_sigma

        self.alpha = alpha
        self.firing_rate_sigma = firing_rate_sigma
        self.low_rank = low_rank
        self.task = task

        self.W_in = nn.Linear(in_dim, hid_dim, bias=True)

        # self.W_rec = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_rec = nn.Parameter(torch.randn(hid_dim, hid_dim))

        if low_rank:
            self.S_l = nn.Parameter(torch.randn(hid_dim, low_rank))
            self.S_r = nn.Parameter(torch.randn(low_rank, hid_dim))

            nn.init.xavier_uniform_(self.S_l)
            nn.init.xavier_uniform_(self.S_r)  

        self.W_out = nn.Linear(hid_dim, out_dim, bias=True)
        
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_rec)
        nn.init.xavier_uniform_(self.W_out.weight)

        # with torch.no_grad():
        #     self.W_rec.weight.fill_diagonal_(0)
    
    def forward(self, inp):
        # inp --> B, T, in_dim
        time = inp.shape[1]
        firing_rates = F.tanh(torch.randn(inp.shape[0], self.hid_dim).to(inp.device) * self.firing_rate_sigma)

        pred_out_list = []
        firing_rate_list = []
        firing_rate_list.append(firing_rates)

        for t in range(time):
            pred, firing_rates = self.forward_step(inp[:, t], firing_rates)

            pred_out_list.append(pred)
            firing_rate_list.append(firing_rates)
        
        return torch.stack(pred_out_list).permute(1, 0, 2), torch.stack(firing_rate_list).permute(1, 0, 2) # B, T, out_dim - B, T, hid_dim
    
    def forward_step(self, inp, firing_rate):
        # input --> B(Batch), D_in(R^{N_{in}})
        # fire_rate --> B(Batch), N+N_2

        # o^(t) = sigmoid(W_out @ r(t))
        if self.task == 'dcdt':
            pred_out = F.tanh(self.W_out(firing_rate)) # B, D_out
        else:
            pred_out = F.sigmoid(self.W_out(firing_rate))

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


# Main code 

# Hyperparameters 


N = 100

input_interval = 0.010
delay_interval = 0.050
reaction_interval = 0.050

delta_t = 5e-3
time_decay = 10e-3
alpha = delta_t / time_decay # 0.5


num_epochs = 3001
task = 'dcdt'
K= None


learning_rate=1e-2
k = 3
seed_everything(seed=k)

rnn = RNN(in_dim=1,
            hid_dim=N,
            out_dim=1,
            alpha=alpha,
            noise_sigma=1e-2,
            firing_rate_sigma=0.1,
            low_rank=K,
            task=task)

data = Single_Dataset(num_of_samples_per_each_class=1,
                        input_interval=input_interval, 
                        delay_interval=delay_interval, 
                        reaction_interval=reaction_interval, 
                        post_reaction_interval=0.,
                        delta_t=delta_t,
                        task=task)

optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

loss_history = []
grad_history = []
val_history = []
acc_history = []

save_epochs = [10,200,1000,3000]
saved_r = {}
saved_acc = {}
state_dicts ={}
out_dicts ={}

for epoch in tqdm(range(num_epochs)):
    inp, gt_out = data.dataset[:, 0], data.dataset[:, 1] # data.dataset.shape = B, (inp, out), T, 1

    optimizer.zero_grad()

    pred_out, r = rnn(inp)
    loss = criterion(pred_out, gt_out)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    acc_history.append(np.mean( binarize(pred_out.cpu().detach().numpy()) == gt_out.cpu().detach().numpy() ) )
    try:
        grad_history.append(torch.norm(rnn.W_rec.grad).cpu().item())
        val_history.append(torch.norm(rnn.W_rec).cpu().item())
    except:
        grad_history.append(torch.norm(rnn.S_l.grad).cpu().item()+torch.norm(rnn.S_r.grad).cpu().item())
    
    #if epoch > 29998:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = 1e-2
    
    if epoch % 1000 == 0:
        plt.subplot(121)
        plt.semilogy(loss_history)
        plt.semilogy(grad_history)
        plt.semilogy(val_history)
        plt.subplot(122)
        
        plt.plot(acc_history)
        plt.show()
        
    if epoch in save_epochs:
        saved_r[epoch] = r.detach().cpu().numpy()
        saved_acc[epoch] = np.mean( binarize(pred_out.cpu().detach().numpy()) == gt_out.cpu().detach().numpy() ) 
        state_dicts[epoch] = {k: v.clone() for k, v in rnn.state_dict().items()}
        out_dicts[epoch] = pred_out.detach().cpu().numpy()
    
#%%

plt.semilogy(np.arange(1,len(loss_history)+1),loss_history)
plt.semilogy(np.arange(1,len(loss_history)+1),grad_history)
for a in save_epochs:
    plt.axvline(x=a+1)
plt.xscale('log')
plt.savefig('figA.pdf')
plt.show()

#%%

plt.plot(acc_history)
for a in save_epochs:
    plt.axvline(x=a+1)
plt.ylim([-0.01,1.01])
plt.xscale('log')
plt.savefig('figB.pdf')
plt.show()


#%%


import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class FixedPointSolver(nn.Module):
    def __init__(self, W: torch.Tensor, b: torch.Tensor, lr=1e-2, max_iters=1000):
        """
        Minimizes L(x) = || -x + tanh(Wx + b) ||_2^2 for x.
        
        Args:
            W: [N, N] weight matrix
            b: [N] bias vector
            lr: learning rate for optimizer
            max_iters: number of gradient steps
        """
        super().__init__()
        self.W = W
        self.b = b
        self.lr = lr
        self.max_iters = max_iters

    def forward(self, x0: torch.Tensor):
        """
        Run optimization starting from initial x0.
        
        Args:
            x0: [B, N] batch of initial guesses
        Returns:
            x_opt: optimized x values [B, N]
            losses: loss history (mean loss per epoch)
        """
        x = nn.Parameter(x0.clone())
        optimizer = optim.Adam([x], lr=self.lr)
        losses = []

        pbar = tqdm(range(self.max_iters))
        for _ in pbar:
            optimizer.zero_grad()
            y = torch.tanh(x @ self.W + self.b)   # [B, N]
            loss = torch.mean(torch.sum((-x + y) ** 2, dim=1))  # mean over batch

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)
            pbar.set_description(f"Loss: {loss_val:.4f}")

        return x.detach(), losses
    
    def compute_speed(self, x: torch.Tensor):
        """Compute per-sample speed: || -x + tanh(Wx+b) ||_2 for each sample."""
        with torch.no_grad():
            y = torch.tanh(x @ self.W + self.b)
            residual = -x + y
            speeds = torch.norm(residual, dim=1)
        return speeds

    def fine_tune_with_hessian(self, x: torch.Tensor, newton_lr: float = 1.0):
        """
        Fine-tune each sample separately with a Newton step using the Hessian.
        
        Args:
            x: [B, N] optimized samples from solver
            newton_lr: step size multiplier for Newton update
        Returns:
            x_refined: [B, N] refined samples
        """
        x_refined = []
        for i in range(x.shape[0]):
            xi = x[i].clone().detach().requires_grad_(True)

            def sample_loss(z):
                y = torch.tanh(z@self.W + self.b)  # [N]
                return torch.sum((-z + y) ** 2)

            # gradient
            grad = torch.autograd.grad(sample_loss(xi), xi, create_graph=True)[0]  # [N]

            # hessian
            H = torch.autograd.functional.hessian(sample_loss, xi)  # [N, N]

            # Newton step: x_new = x - H^{-1} g
            try:
                dx = torch.linalg.solve(H + 1e-4*torch.eye(H.shape[0]), grad)  # regularized inverse
                xi_new = xi - newton_lr * dx
            except RuntimeError:
                # fallback if Hessian is singular
                xi_new = xi - newton_lr * grad * 0.01

            x_refined.append(xi_new.detach())

        return torch.stack(x_refined, dim=0)


def jacobian_at_fp(W,b, x_fp):
    """
    Jacobian of the RNN update at fixed point x_fp.
    W: (N, N) weight matrix, as in  W @ r 
    x_fp: (N,) fixed point (1D tensor)
    alpha: float
    """
    with torch.no_grad():
        z = W @ x_fp    +b      # (N,)
        phi_prime = 1.0 - torch.tanh(z)**2   # (N,)
        J = -torch.eye(W.shape[0]) \
            + torch.diag(phi_prime) @ W
    return J
    
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_pca_trajectories(saved_r_dict, saved_acc, saved_dicts, max_iters=5000,it_vals=None):
    # stack everything: list of [num_trials, T, N]
    all_r = np.concatenate(
        [r_val.reshape(-1, r_val.shape[-1]) for r_val in saved_r_dict.values()],
        axis=0
    )  # shape: [total_samples, N]

    if it_vals == None:
        save = 1
    else:
        save = 0

    if it_vals is None:
        it_vals = range(1,len(saved_r_dict)+1)

    # fit PCA once on ALL data
    pca = PCA(n_components=5)
    pca.fit(all_r)
    print("Explained variance ratio:", pca.explained_variance_ratio_[:3])

    fig = plt.figure(figsize=(5 * len(saved_r_dict), 6))

    for i, (epoch, r_val) in enumerate(saved_r_dict.items(), start=1):
        if i in it_vals:
            
        
            ax = fig.add_subplot(1, len(saved_r_dict), i, projection='3d')
    
            num_trials, T, N = r_val.shape
            # ---- trajectories ----
            for trial in range(num_trials):
                traj = r_val[trial]  # [T, N]
                traj_3d = pca.transform(traj)[:, :3]  # keep first 3 PCs
    
                label = f'Match {trial+1}' if trial < 2 else f'No-Match {trial+1}'
                color = plt.rcParams["axes.prop_cycle"].by_key()["color"][trial]
    
                ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2],
                        marker='.', markersize=3, label=label, color=color, alpha=0.8)
    
                # mark start and end
                #ax.scatter(traj_3d[0, 0], traj_3d[0, 1], traj_3d[0, 2],
                #           c='green', s=50, marker='x')
                #ax.scatter(traj_3d[-1, 0], traj_3d[-1, 1], traj_3d[-1, 2],
                #           c='red', s=50, marker='o')
    
            # ---- fixed points (first-order + Newton refinement) ----
            cur_rnn = saved_dicts[epoch]
            W = torch.tensor(cur_rnn['W_rec'].detach().cpu().numpy())
            b = torch.tensor(cur_rnn['W_in.bias'].detach().cpu().numpy())
    
    
            
            # flatten r values for initialization
            x0 = torch.tensor(r_val.reshape(-1, r_val.shape[-1]))
            
            solver = FixedPointSolver(W, b, lr=1e-2, max_iters=max_iters)
            x_opt, _ = solver(x0)
            
            # === Newton refinement ===
            x_refined = solver.fine_tune_with_hessian(x_opt, newton_lr=1.0)
            
            print("Speed difference:", torch.min(solver.compute_speed(x_opt)-solver.compute_speed(x_refined)))
            
            # ---- find closest to initial conditions ----
            dists = torch.cdist(x_refined, x0)   # shape: [num_unique, num_inits]
            min_dists = dists.min(dim=1).values # [num_unique]
            
            # ---- project into PCA space ----
            x_proj_all = pca.transform(x_refined.numpy())[:, :3]
    
            
            # round PCA coords to identify overlaps
            x_proj_rounded = np.round(x_proj_all, decimals=1)
            
    
            
            # keep track of chosen indices
            keep_indices = []
            seen = {}
            
            for idx, coord in enumerate(map(tuple, x_proj_rounded)):
                d = float(min_dists[idx].item())
                #if d<10:
                if coord not in seen:
                    seen[coord] = (idx, d)  # first time seeing this coord
                else:
                    prev_idx, prev_d = seen[coord]
                    # keep the one closer to init
                    if d < prev_d:
                        seen[coord] = (idx, d)
                        
            
            # collect final unique indices
            keep_indices = [idx for idx, _ in seen.values()]
            
            # final selection
            x_closest = x_refined[keep_indices]
            dists_closest = min_dists[keep_indices]
            speeds = (solver.compute_speed(x_closest))
            x_proj = x_proj_all[keep_indices]
            
            # ---- plotting ----
            sc = ax.scatter(
                x_proj[:, 0], x_proj[:, 1], x_proj[:, 2],
                c=np.log10(speeds.numpy()), cmap='coolwarm',edgecolors='none',
                marker='D', s=40, vmin = -6, vmax = 0, depthshade=False
            )
            
            cbar = fig.colorbar(
                sc,
                orientation="horizontal",
                location="bottom",
                fraction=0.05, pad=0.15     # tweak size/spacing
            )
            cbar.set_label("log10(Speed)")
            
            
            
            # Change the line below to study the properties of the slow points
            idx_closest = torch.argsort(dists_closest)[0]
            
            
            
            fp_slowest = x_closest[idx_closest]
            J = jacobian_at_fp(W.T,b, fp_slowest)
            eigvals = torch.linalg.eigvals(J).cpu().numpy()
            print("distance:",dists_closest[idx_closest])
            print("Speed:", (speeds[idx_closest]))
            print("Eigvals:",np.sort(np.real(eigvals)))
    
            
            # annotate with distance to init
            for (x, y, z, d) in zip(x_proj[:, 0], x_proj[:, 1], x_proj[:, 2], dists_closest.numpy()):
                ax.text(x + 0.5, y + 0.1, z, f"{d:.2f}", fontsize=8)
                
            #ax.scatter(x_proj[idx_closest, 0], x_proj[idx_closest, 1], x_proj[idx_closest, 2],
            #   edgecolors='none', marker='x', s=40, vmin = -3, vmax = 0
            #)
            
            
            ax.set_title(f"Epoch {epoch+1}\nacc {saved_acc[epoch]:.3f}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_xlim([-5, 7])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-3, 3])
            ax.legend()
            ax.grid(False)
            # make panes transparent (no background)
            ax.xaxis.pane.set_visible(False)
            ax.yaxis.pane.set_visible(False)
            ax.zaxis.pane.set_visible(False)
            
           
        
    # add shared colorbar for speeds
    #fig.colorbar(sc, ax=fig.get_axes(), shrink=0.6, label="Speed")

    plt.tight_layout()
    if save:
        plt.savefig('figC.pdf')
    plt.show()

plot_pca_trajectories(saved_r,saved_acc,state_dicts,it_vals = None)
        
        
        