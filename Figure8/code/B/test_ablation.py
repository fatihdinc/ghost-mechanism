import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import copy
from tqdm import tqdm

class RankOneRNN(nn.Module):
    def __init__(self, N, alpha,c):
        super(RankOneRNN, self).__init__()
        self.N = N
        self.alpha = alpha
        self.m = nn.Parameter(torch.randn(N) / np.sqrt(N))
        self.n = nn.Parameter(torch.randn(N) / np.sqrt(N))
        self.b = nn.Parameter(torch.zeros(N))
        self.noise_std = 0.0
        self.c = c

    def forward(self, x):
        noise = torch.randn(self.N) * self.noise_std
        kappa = torch.dot(self.n, x)
        phi = torch.tanh(self.m * kappa + self.b + noise)
        x = (1 - self.alpha) * x + self.alpha * phi
        kappa = torch.dot(self.n, x)
        return x, kappa


def generate_target(T_disc, total_time):
    return torch.cat([torch.zeros(T_disc), torch.ones(total_time - T_disc)])


def train_rnn(rnn, T, T_disc, total_time, learning_rate, num_epochs, plot_interval, rnn_type, factor):
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    target = generate_target(T_disc, total_time)

    

    loss_history = []
    output_history = []
    dkappa_vs_kappa_history = []
    grad_history = []
    acc_history = []

    for epoch in range(num_epochs):
        x = torch.randn(rnn.N)/10
        x = x*0;
        x = x - x.mean()-3/10
        optimizer.zero_grad()

        output = []
        x_epoch = x.clone()

        for t in range(total_time):
            x_epoch, kappa = rnn(x_epoch)
            output.append(torch.sigmoid(rnn.c* (kappa - 1)))

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

        if epoch % plot_interval == 0:
            accuracy = ((output > 0.5).float() == target).float().mean().item()
            acc_history.append(accuracy)
            #print(f"{rnn_type} RNN - Epoch {epoch}, Loss: {loss.item()}")
            
            #print(f"{rnn_type} RNN - Accuracy: {accuracy:.4f}")

            kappa_range = torch.linspace(-15, 15, 1000)
            dkappa_values = []
            for kappa in kappa_range:
                phi = torch.tanh(rnn.m * kappa + rnn.b)
                dkappa = -kappa + torch.dot(rnn.n, phi)
                dkappa_values.append(dkappa.item())
            dkappa_vs_kappa_history.append((kappa_range.numpy(), dkappa_values))
        #if epoch % 100 == 0:
            #print(f"{rnn_type} RNN - Epoch {epoch}, Loss: {loss.item()}")
    return rnn, loss_history, acc_history, output_history, dkappa_vs_kappa_history,grad_history

# Main code 
import random

# Hyperparameters 
N = 100
alpha = 0.5
T = 20
T_disc = int(T)
total_time = 2 * T_disc
learning_rate = .02
c_all = np.logspace(-3,1,30)

acc_vals = np.zeros([100,30,6100])
grad_vals = np.zeros([100,30,6100])

for exp in tqdm(range(100)):
    # Set random seed for reproducibility
    seed = exp
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    
    
    
    num_epochs = 6000
    plot_interval = 1
    current_time = datetime.now().time()
    
    
    
    # Train the original RankOneRNN
    original_rnn = RankOneRNN(N, alpha,10)
    original_rnn, original_loss_history, acc_history, _, original_dkappa_vs_kappa_history,\
        grad_history = train_rnn(original_rnn, 
                                 T, T_disc, total_time, 
                                 learning_rate, num_epochs, 
                                 plot_interval, "Original", "original")
    
    acc_vals[exp,:,:6000]= acc_history
    grad_vals[exp,:,:6000] =grad_history

    num_epochs = 100
    
    
    for k in range(30):
        
        rnn = RankOneRNN(N, alpha,10)
        
        rnn.load_state_dict(copy.deepcopy(original_rnn.state_dict()))
        
        rnn.c =c_all[k];
        rnn, loss, acc_history, _, original_dkappa_vs_kappa_history,\
            grad_history = train_rnn(rnn, 
                                     T, T_disc, total_time, 
                                     learning_rate, num_epochs, 
                                     plot_interval, "Original", "original")
        acc_vals[exp,k,6000:] = acc_history
        grad_vals[exp,k,6000:] =grad_history
    
    
    
np.savez('confidence_ablation.npz',acc_vals = acc_vals,c_all=c_all,grad_vals = grad_vals)



