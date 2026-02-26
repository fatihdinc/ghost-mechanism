import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import copy

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
        noise = torch.randn(N) * self.noise_std
        kappa = torch.dot(self.n, x)
        phi = torch.tanh(self.m * kappa + self.b + noise)
        x = (1 - self.alpha) * x + self.alpha * phi
        kappa = torch.dot(self.n, x)
        return x, kappa


class RankOneRNN_conf(nn.Module):
    def __init__(self, N, alpha):
        super(RankOneRNN_conf, self).__init__()
        self.N = N
        self.alpha = alpha
        self.m = nn.Parameter(torch.randn(N) / np.sqrt(N))
        self.n = nn.Parameter(torch.randn(N) / np.sqrt(N))
        self.b = nn.Parameter(torch.zeros(N))
        self.noise_std = 0.0
        self.c = nn.Parameter(torch.zeros(1)+10)

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

def train_rnn(rnn, T, T_disc, total_time, learning_rate, num_epochs, plot_interval, rnn_type, factor):
    optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)
    target = generate_target(T_disc, total_time)

    

    loss_history = []
    b_new_history = []
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
        if epoch % 100 == 0:
            print(f"{rnn_type} RNN - Epoch {epoch}, Loss: {loss.item()}")
    return rnn, loss_history, acc_history, output_history, dkappa_vs_kappa_history,grad_history

def plot_fig(original_dkappa_vs_kappa_history,lr):
    # Plot 1: Epoch-wise dk/dt vs k subplots 
    total_subplots = len(original_dkappa_vs_kappa_history) 
    subplot_cols = 5
    subplot_rows = (total_subplots - 1) // subplot_cols + 1
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 4 * subplot_rows))
    axs = axs.flatten()

    for i, (kappa, dkappa) in enumerate(original_dkappa_vs_kappa_history):
        axs[i].plot(kappa, dkappa, color='blue')
        axs[i].set_title(f'Original Epoch {i * plot_interval}')
        axs[i].set_xlabel('κ')
        axs[i].set_ylabel('dκ/dt')
        axs[i].axhline(y=0, color='r', linestyle='--')
        axs[i].axvline(x=0, color='r', linestyle='--')
        axs[i].grid(True)


    for i in range(total_subplots, len(axs)):
        axs[i].axis('off')

    fig.suptitle('Epoch-wise dk/dt vs k', fontsize=16)
    plt.tight_layout()
    plt.savefig('epoch_k_dk_lr_%.f.pdf' %(lr), dpi=300)
    plt.show()
    plt.close(fig)


# Main code 
import random

# Hyperparameters 
N = 100
alpha = 0.5
T = 20
T_disc = int(T)
total_time = 2 * T_disc
learning_rate = .02

seed = 40
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
    


num_epochs = 10000
plot_interval = 1
current_time = datetime.now().time()

print('Start time:',current_time.strftime("%H:%M:%S"))

# Train the original RankOneRNN
original_rnn = RankOneRNN_conf(N, alpha)
original_rnn, original_loss_history, acc_history, _, original_dkappa_vs_kappa_history,\
    grad_history = train_rnn(original_rnn, 
                             T, T_disc, total_time, 
                             learning_rate, num_epochs, 
                             plot_interval, "Original", "original")


#%%


conf_rnn = RankOneRNN_conf(N, alpha)

conf_rnn.m.data = original_rnn.m.data
conf_rnn.n.data = original_rnn.n.data
conf_rnn.b.data = original_rnn.b.data
conf_rnn.c.data = torch.tensor(1,dtype = torch.float32)

conf_rnn_new, loss_new, acc_history_new, _, dkappa_kappa_new,\
    grad_new = train_rnn(conf_rnn, 
                             T, T_disc, total_time, 
                             learning_rate, num_epochs, 
                             plot_interval, "Original", "original")

#%%
plt.semilogy(np.append(original_loss_history,loss_new))
plt.semilogy(np.append(grad_history,grad_new),alpha = 0.5)
plt.savefig('learnable_confidence.pdf')
plt.show()

    