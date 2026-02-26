import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

class RankOneRNN(nn.Module):
    def __init__(self, N, alpha):
        super(RankOneRNN, self).__init__()
        self.N = N
        self.alpha = alpha
        self.m = nn.Parameter(torch.randn(N) / np.sqrt(N))
        self.n = nn.Parameter(torch.randn(N) / np.sqrt(N))
        self.b = nn.Parameter(torch.zeros(N))
        self.noise_std = 0.0

    def forward(self, x):
        noise = torch.randn(N) * self.noise_std
        kappa = torch.dot(self.n, x)
        phi = torch.tanh(self.m * kappa + self.b + noise)
        x = (1 - self.alpha) * x + self.alpha * phi
        kappa = torch.dot(self.n, x)
        return x, kappa


def generate_target(T_disc, total_time):
    return torch.cat([torch.zeros(T_disc), torch.ones(total_time - T_disc)])

def find_saddle_point(rnn, target_neuron, factor):
    def calculate_dkappa(kappa):
        phi = torch.tanh(rnn.m * kappa + rnn.b)
        dkappa = -kappa + torch.dot(rnn.n, phi)
        return dkappa.item()

    kappa_range = torch.linspace(-5, 5, 1000)
    dkappa_values = [calculate_dkappa(k) for k in kappa_range]
    
    # Find the right peak
    peak_index = np.argmax(dkappa_values[len(dkappa_values)//2:]) + len(dkappa_values)//2
    peak_kappa = kappa_range[peak_index].item()
    
    # Find the zero crossing to the right of the peak
    zero_crossings = np.where(np.diff(np.sign(dkappa_values[peak_index:])))[0]
    if len(zero_crossings) == 0:
        print("Couldn't find a zero crossing to the right of the peak.")
        return 0
    
    zero_crossing_index = zero_crossings[0] + peak_index
    zero_crossing_kappa = kappa_range[zero_crossing_index].item()
    
    # Set b_new
    b_new_value = -factor * zero_crossing_kappa
    rnn.b_new.data[target_neuron] = b_new_value
    
    print(f"Found right peak at κ = {peak_kappa}")
    print(f"Found right zero crossing at κ = {zero_crossing_kappa}")
    print(f"Setting b_new[{target_neuron}] to {b_new_value}")
    
    # Plot for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(kappa_range.numpy(), dkappa_values)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=peak_kappa, color='g', linestyle='--', label='Right peak')
    plt.axvline(x=zero_crossing_kappa, color='b', linestyle='--', label='Right zero crossing')
    plt.scatter([peak_kappa], [max(dkappa_values[len(dkappa_values)//2:])], color='g', s=100, zorder=5, label='Peak')
    plt.scatter([zero_crossing_kappa], [0], color='b', s=100, zorder=5, label='Zero crossing')
    plt.xlabel('κ')
    plt.ylabel('dκ/dt')
    plt.title('dκ/dt vs κ with right peak and zero crossing')
    plt.legend()
    plt.grid(True)
    plt.savefig('dkappa_vs_kappa_with_peak_and_zero.pdf')
    plt.close()
    
    return b_new_value

def plot_dkappa_vs_kappa(rnn, factor, ax):
    with torch.no_grad():
        kappa_range = torch.linspace(-5, 5, 1000)
        dkappa_values = []

        for kappa in kappa_range:
            phi = torch.tanh(rnn.m * kappa + rnn.b)
            phi += rnn.alpha * rnn.b_new
            dkappa = -kappa + torch.dot(rnn.n, phi)
            dkappa_values.append(dkappa.item())

        ax.plot(kappa_range.numpy(), dkappa_values)
        ax.set_xlabel('κ')
        ax.set_ylabel('dκ/dt')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.grid(True)

def train_rnn(rnn, T, T_disc, total_time, learning_rate, num_epochs, rnn_type, factor):
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    target = generate_target(T_disc, total_time)

    

    loss_history = []
    b_new_history = []
    output_history = []
    dkappa_vs_kappa_history = []
    grad_history = []

    for epoch in range(num_epochs):
        x = torch.randn(rnn.N)/10
        x = x*0;
        x = x - x.mean()-3/10
        optimizer.zero_grad()

        output = []
        x_epoch = x.clone()

        for t in range(total_time):
            x_epoch, kappa = rnn(x_epoch)
            output.append(torch.sigmoid(10* (kappa - 1)))

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

# Set random seed for reproducibility
seed = 40

# Hyperparameters 
N = 100
alpha = 0.5
T = 20
T_disc = int(T)
total_time = 2 * T_disc
num_epochs = 200001


lr_all = np.logspace(-4,-1,30)

loss_all = np.zeros([10,30,num_epochs])
grad_all = np.zeros([10,30,num_epochs])

for i in range(30):
    current_time = datetime.now().time()
    print(i,'started. start time:',current_time.strftime("%H:%M:%S"), '\n')
    learning_rate = lr_all[i]
    
    for k in range(10):
        seed = k;
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Train the original RankOneRNN
        original_rnn = RankOneRNN(N, alpha)
        original_rnn, original_loss_history, _, _, original_dkappa_vs_kappa_history,grad_history = train_rnn(original_rnn, T, T_disc, total_time, learning_rate, num_epochs, "Original", "original")
        loss_all[k,i,:] = original_loss_history[:num_epochs];
        grad_all[k,i,:] = grad_history[:num_epochs]


np.savez('loss_save.npz',loss_all = loss_all,grad_all=grad_all)

