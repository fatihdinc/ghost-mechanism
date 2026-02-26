#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 06:02:48 2026

@author: fatih
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

import os

class RankKRNN(nn.Module):
    def __init__(self, N, alpha,K=1):
        super(RankKRNN, self).__init__()
        self.N = N
        self.alpha = alpha
        self.m = nn.Parameter(torch.randn(N,K) / np.sqrt(N))
        self.n = nn.Parameter(torch.randn(K,N) / np.sqrt(N))
        self.b = nn.Parameter(torch.zeros(N))
        self.W_out = nn.Parameter(torch.randn(N) / np.sqrt(N))
        self.b_out = nn.Parameter(torch.randn(1) / np.sqrt(N))
        self.noise_std = 0.0

    def forward(self, x):
        phi = torch.tanh(x@self.m @self.n + self.b)
        x = (1 - self.alpha) * x + self.alpha * phi
        out = torch.sigmoid(x @ self.W_out + self.b_out)
        return x,out


os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)


def generate_target(T_disc, total_time):
    return torch.cat([torch.zeros(T_disc), torch.ones(total_time - T_disc)])




def train_rnn(rnn, T, T_disc, total_time, learning_rate, num_epochs, rnn_type, factor,seed, K):
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    target = generate_target(T_disc, total_time)

    

    loss_history = []
    b_new_history = []
    output_history = []
    dkappa_vs_kappa_history = []
    grad_history = []
    
    pbar = range(num_epochs)

    for epoch in pbar:
        x = torch.randn(rnn.N)/10
        x = x*0;
        x = x - x.mean()-3/10
        optimizer.zero_grad()

        output = []
        x_epoch = x.clone()

        for t in range(total_time):
            x_epoch, out = rnn(x_epoch)
            output.append(out)

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
        
        
    
    

    save_path = f"results/results_seed{seed}_K{K}_lr{learning_rate:.2e}.npz"

    np.savez(
        save_path,
        loss=np.array(loss_history, dtype=np.float32),
        grad=np.array(grad_history, dtype=np.float32)
    )

    plt.figure()
    plt.loglog(np.array(loss_history))
    plt.loglog(np.array(grad_history),alpha=0.5)
    plot_path = f"plots/plot_seed{seed}_K{K}_lr{learning_rate:.2e}.pdf"
    plt.savefig(plot_path)
    plt.show()

        
    return rnn, loss_history, b_new_history, output_history,dkappa_vs_kappa_history,grad_history


# Main code 
import random


def run_single(seed, lr, N, alpha, T, T_disc, total_time, num_epochs,K):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize model
    rnn = RankKRNN(N, alpha,K)

    # Train
    _, loss_history, _, _, _, grad_history = train_rnn(
    rnn,
    T=T,
    T_disc=T_disc,
    total_time=total_time,
    learning_rate=lr,
    num_epochs=num_epochs,
    rnn_type="Original",
    factor="original",
    seed=seed,
    K=K
    )


    return seed, lr, K, np.array(loss_history), np.array(grad_history)


# Hyperparameters
N = 100
alpha = 0.5
T = 20
T_disc = int(T)
total_time = 2 * T_disc
num_epochs = 500000

seeds = list(range(10))
lr_all = np.logspace(-3,0,10)
K_all = np.array([1,2,3,4,5,10,30,50,100]).astype(int)


n_seeds = len(seeds)
n_lrs = len(lr_all)
n_K = len(K_all)

combos = [
    (seed, lr, K)
    for lr in lr_all
    for seed in seeds
    for K in K_all
]

results = Parallel(n_jobs=6)(
    delayed(run_single)(
        seed, lr,
        N, alpha, T, T_disc, total_time, num_epochs, K
    )
    for (seed, lr, K) in tqdm(
        combos,
        total=len(combos),
        desc="Running experiments"
    )
)






