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



def train_rnn(rnn, T, T_disc, total_time, learning_rate, num_epochs, plot_interval, rnn_type, factor):
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
        optimizer.step()

        loss_history.append(loss.item())
        output_history.append(output.detach().numpy())
        
        
        l2_norm = 0.0
        for param in rnn.parameters():
            if param.grad is not None:
                l2_norm += param.grad.norm(2).item() ** 2
        
        l2_norm = l2_norm ** 0.5 
        grad_history.append(l2_norm)

        if epoch % plot_interval == 0:
            accuracy = ((output > 0.5).float() == target).float().mean().item()
            print(f"{rnn_type} RNN - Epoch {epoch}, Loss: {loss.item()}")
            print(f"{rnn_type} RNN - Accuracy: {accuracy:.4f}")

            kappa_range = torch.linspace(-15, 15, 1000)
            dkappa_values = []
            for kappa in kappa_range:
                phi = torch.tanh(rnn.m * kappa + rnn.b)
                dkappa = -kappa + torch.dot(rnn.n, phi)
                dkappa_values.append(dkappa.item())
            dkappa_vs_kappa_history.append((kappa_range.numpy(), dkappa_values))
    
    return rnn, loss_history, b_new_history, output_history, dkappa_vs_kappa_history,grad_history



# Main code 
import random

# Set random seed for reproducibility
seed = 40
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Hyperparameters 
N = 100
alpha = 0.5
T = 20
T_disc = int(T)
total_time = 2 * T_disc
learning_rate = .003
num_epochs = 500000
plot_interval = 1000
current_time = datetime.now().time()

print('Start time:',current_time.strftime("%H:%M:%S"))

# Train the original RankOneRNN
original_rnn = RankOneRNN(N, alpha)
original_rnn, original_loss_history, _, _, original_dkappa_vs_kappa_history,grad_history = train_rnn(original_rnn, T, T_disc, total_time, learning_rate, num_epochs, plot_interval, "Original", "original")

#%%

plt.subplot(211)
plt.semilogy(grad_history)

plt.subplot(212)
plt.semilogy(original_loss_history)
plt.show()

#%%
kappa_all = np.zeros(100)

for i in range(100):

    x = torch.randn(original_rnn.N)/10*0
    x = x - x.mean()-3/10
    
    kappa_in = torch.dot(original_rnn.n, x)
    kappa_all[i] = kappa_in.detach().numpy()

#%%

x = original_dkappa_vs_kappa_history[0]
y = original_dkappa_vs_kappa_history[-1]

#plt.plot(x[0],x[1])
plt.plot(y[0][200:800],y[1][200:800])
plt.scatter(kappa_all,np.zeros(kappa_all.shape),50,'red','.')
plt.axhline(0,color = 'black',ls = '--')
plt.axvline(1,color = 'orange',ls = '--')
plt.xlabel('kappa')
plt.ylabel('dkappa/dt')


np.savez('long_training.npz',original_loss_history=original_loss_history,
         original_dkappa_vs_kappa_history=original_dkappa_vs_kappa_history,
         grad_history=grad_history,kappa_all=kappa_all)

