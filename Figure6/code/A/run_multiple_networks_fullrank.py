import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed

class RNN(nn.Module):
    def __init__(self, N, alpha):
        super(RNN, self).__init__()
        self.N = N
        self.alpha = alpha
        self.m = nn.Parameter(torch.randn(N,N) / np.sqrt(N))
        self.b = nn.Parameter(torch.zeros(N))
        self.W_out = nn.Parameter(torch.randn(N) / np.sqrt(N))
        self.b_out = nn.Parameter(torch.randn(1) / np.sqrt(N))
        self.noise_std = 0.0

    def forward(self, x):
        phi = torch.tanh(x@self.m + self.b)
        x = (1 - self.alpha) * x + self.alpha * phi
        out = torch.sigmoid(x @ self.W_out + self.b_out)
        return x,out


def generate_target(T_disc, total_time):
    return torch.cat([torch.zeros(T_disc), torch.ones(total_time - T_disc)])



def train_rnn(rnn, T, T_disc, total_time, learning_rate, num_epochs, rnn_type, factor):
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    target = generate_target(T_disc, total_time)

    

    loss_history = []
    b_new_history = []
    output_history = []
    dkappa_vs_kappa_history = []
    grad_history = []

    # for epoch in tqdm(range(num_epochs), desc="Training epochs", leave=False):
    for epoch in range(num_epochs):
        x = torch.randn(rnn.N)/10
        x = x*0;
        x = x - x.mean()-3/10
        optimizer.zero_grad()

        output = []
        x_epoch = x.clone()

        for t in range(total_time):
            x_epoch, kappa = rnn(x_epoch)
            output.append(kappa)

        output = torch.stack(output).squeeze()

        loss = torch.sum((output - target)**2) / T
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(loss.item())
        output_history.append(output.detach().numpy())
        # Step 6: Compute the L2 norm of the gradient
        l2_norm = 0.0
        for param in rnn.parameters():
            if param.grad is not None:
                l2_norm += param.grad.norm(2).item() ** 2
        
        l2_norm = l2_norm ** 0.5  # Take the square root to get the L2 norm
        grad_history.append(l2_norm)

    
    return rnn, loss_history, b_new_history, output_history, dkappa_vs_kappa_history,grad_history

# Main code 
import random

n_processes=8
# Set random seed for reproducibility
seed = 40

# Hyperparameters 
N = 100
alpha = 0.5
T = 20
T_disc = int(T)
total_time = 2 * T_disc
num_epochs = 200001
n_exp = 10

lr_all = np.logspace(-4,1,30)

loss_all = np.zeros([n_exp,lr_all.shape[0],num_epochs])
grad_all = np.zeros([n_exp,lr_all.shape[0],num_epochs])

def run_training(i, k, learning_rate):
    seed = k
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rnn = RNN(N, alpha)
    return (i, k, *train_rnn(rnn, T, T_disc, total_time, learning_rate, num_epochs, "Original", "original"))

# Prepare all jobs: 30 learning rates × 10 seeds
all_jobs = [(i, k, lr_all[i]) for i in range(len(lr_all)) for k in range(n_exp)]

# Run all jobs in parallel
results = Parallel(n_jobs=n_processes)(
    delayed(run_training)(i, k, lr) for (i, k, lr) in tqdm(all_jobs, desc="All training jobs")
)

# Fill results into arrays
for i, k, rnn_model, loss_history, _, _, _, grad_history in results:
    loss_all[k, i, :] = loss_history[:num_epochs]
    grad_all[k, i, :] = grad_history[:num_epochs]

np.savez('loss_save_full.npz',loss_all = loss_all,grad_all=grad_all)

