import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

from datetime import datetime


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
        firing_rates = F.tanh(torch.randn(inp.shape[0], self.hid_dim, requires_grad=True).to(inp.device) * self.firing_rate_sigma)

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

input_interval = 0.030
delay_interval = 0.080
reaction_interval = 0.050

delta_t = 5e-3
time_decay = 10e-3
alpha = delta_t / time_decay # 0.5


num_epochs = 6000
task = 'dms'

K= None
learning_rate=1.438449888287663
k = 0
seed_everything(seed=k)

rnn = RNN(in_dim=1,
            hid_dim=N,
            out_dim=1,
            alpha=alpha,
            noise_sigma=0,
            firing_rate_sigma=0,
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



for epoch in tqdm(range(num_epochs)):
    inp, gt_out = data.dataset[:, 0], data.dataset[:, 1] # data.dataset.shape = B, (inp, out), T, 1

    optimizer.zero_grad()

    pred_out, fr = rnn(inp)
    loss = criterion(pred_out, gt_out)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    grad_history.append(torch.norm(rnn.W_rec.grad).cpu().item())
    val_history.append(torch.norm(rnn.W_rec).cpu().item())
    
    if epoch == 5750:
        fr_beforem2 = fr[0,:,:].detach().numpy()
        out_beforem2 = pred_out[0,:,:].detach().numpy().flatten()
        grad_beforem2 = torch.norm(rnn.W_rec.grad).cpu().item()
        wout_beforem2 = rnn.W_out.weight.cpu().detach()
        bout_beforem2 = rnn.W_out.bias.item()
        
        
        
    if epoch == 5751:
        fr_before = fr[0,:,:].detach().numpy()
        out_before = pred_out[0,:,:].detach().numpy().flatten()
        grad_before = torch.norm(rnn.W_rec.grad).cpu().item()
        wout_before = rnn.W_out.weight.cpu().detach()
        bout_before = rnn.W_out.bias.item()
        
    if epoch == 5752:
        fr_after = fr[0,:,:].detach().numpy()
        out_after = pred_out[0,:,:].detach().numpy().flatten()
        grad_after = torch.norm(rnn.W_rec.grad).cpu().item()
        wout_after = rnn.W_out.weight.cpu().detach()
        bout_after =rnn.W_out.bias.item()


plt.semilogy(loss_history)
plt.semilogy(grad_history)
plt.semilogy(val_history)
plt.xlim([5740,5760])
plt.axvline(x=5750)
plt.axvline(x=5751)
plt.axvline(x=5752)
plt.savefig('loss.pdf')
plt.show()        

#%%
#plt.hist(fr_beforem2.flatten(),bins = np.linspace(-1,1,30))
plt.hist(fr_before.flatten(),alpha =0.5,bins = np.linspace(-1,1,30))
plt.hist(fr_after.flatten(),alpha =0.3,bins = np.linspace(-1,1,30))
plt.savefig('firing_rates.pdf')
plt.show()



plt.plot(gt_out[0,:,0].detach().numpy(),color = 'black',ls = '--')
#plt.plot(out_beforem2[:38])
plt.plot(out_before[:38])
plt.plot(out_after[:38])
plt.savefig('outputs.pdf')
plt.show()






