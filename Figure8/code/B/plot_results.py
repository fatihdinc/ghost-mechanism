#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 18:52:04 2025

@author: dinc
"""



import numpy as np
import matplotlib.pyplot as plt

f = np.load('confidence_ablation.npz')
acc_vals = f['acc_vals']
c_all = f['c_all']

x_permuted = np.transpose(acc_vals, (0, 2, 1))

c_all2 = np.log10(c_all)

ind1 = np.where( (np.max(x_permuted[:,5950:6000,0],1)<=0.5) )[0]
ind2 = np.where( ( np.max(x_permuted[:,:6000,0],1)>0.5 ) )[0]
ind = np.intersect1d(ind1,ind2)


x_permuted = x_permuted[ind,:,:]# Calculate the mean and standard error of the mean (SEM) across runs
mean_x = np.mean(x_permuted, axis=0)
sem_x = np.std(x_permuted, axis=0) / np.sqrt(x_permuted.shape[0])

mean_x = mean_x[:,2::3]
sem_x = sem_x[:,2::3]
c_all2 = c_all2[2::3]

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Generate colors for the last 100 epochs
norm = plt.Normalize(vmin=min(c_all2), vmax=max(c_all2))
colors = plt.cm.viridis_r(norm(c_all2))

# Plot the first 6000 epochs in black with error bars every 10 values
for param_index in range(mean_x.shape[1]):
    ax.plot(range(6000), mean_x[:6000, param_index], color='black', alpha=0.7)
    ax.errorbar(range(0, 6000, 10), mean_x[:6000:10, param_index], yerr=sem_x[:6000:10, param_index], 
                fmt='none', ecolor='black', alpha=0.7)

# Plot the last 100 epochs in varying colors with error bars every 10 values
for param_index in range(mean_x.shape[1]):
    ax.plot(range(6000, 6100), mean_x[6000:, param_index], color=colors[param_index],label = f'{10**c_all2[param_index]}')
    ax.errorbar(range(6000+param_index, 6100, 10), mean_x[6000+param_index::10, param_index], yerr=sem_x[6000+param_index::10, param_index], 
                fmt='none', ecolor=colors[param_index])

# Set labels and title
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Only after the confidence is lowered, can the RNN recover from the no learning zone.')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
sm.set_array([])
#cbar = plt.colorbar(sm, ax=ax)
#cbar.set_label('Log confidence level')
plt.axvline(6000,color = 'red',ls = '--')
plt.legend()

plt.xlim([5800,6100])

# Show the plot
plt.tight_layout()
plt.savefig('confidence_acc_plots.pdf')
plt.show()



#%%



import numpy as np
import matplotlib.pyplot as plt

f = np.load('confidence_ablation.npz')
acc_vals = f['acc_vals']
c_all = f['c_all']
grad_vals = f['grad_vals']

x_permuted = np.transpose(acc_vals, (0, 2, 1))

c_all2 = np.log10(c_all)

ind1 = np.where( (np.max(x_permuted[:,5950:6000,0],1)<=0.5) )[0]
ind2 = np.where( ( np.max(x_permuted[:,:6000,0],1)>0.5 ) )[0]
ind = np.intersect1d(ind1,ind2)


x_permuted = np.transpose(grad_vals, (0, 2, 1))

x_permuted = x_permuted[ind,:,:]# Calculate the mean and standard error of the mean (SEM) across runs
mean_x = np.mean(x_permuted, axis=0)
sem_x = np.std(x_permuted, axis=0) / np.sqrt(x_permuted.shape[0])

mean_x = mean_x[:,2::3]
sem_x = sem_x[:,2::3]
c_all2 = c_all2[2::3]

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Generate colors for the last 100 epochs
norm = plt.Normalize(vmin=min(c_all2), vmax=max(c_all2))
colors = plt.cm.viridis_r(norm(c_all2))

# Plot the first 6000 epochs in black with error bars every 10 values
for param_index in range(mean_x.shape[1]):
    ax.plot(range(6000), mean_x[:6000, param_index], color='black', alpha=0.7)
    ax.errorbar(range(0, 6000, 10), mean_x[:6000:10, param_index], yerr=sem_x[:6000:10, param_index], 
                fmt='none', ecolor='black', alpha=0.7)

# Plot the last 100 epochs in varying colors with error bars every 10 values
for param_index in range(mean_x.shape[1]):
    ax.plot(range(6000, 6100), mean_x[6000:, param_index], color=colors[param_index],label = f'{10**c_all2[param_index]}')
    ax.errorbar(range(6000+param_index, 6100, 10), mean_x[6000+param_index::10, param_index], yerr=sem_x[6000+param_index::10, param_index], 
                fmt='none', ecolor=colors[param_index])

# Set labels and title
ax.set_xlabel('Number of epochs')
ax.set_ylabel('Gradient')
ax.set_title('Only after the confidence is lowered, can the RNN recover from the no learning zone.')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
sm.set_array([])
#cbar = plt.colorbar(sm, ax=ax)
#cbar.set_label('Log confidence level')
plt.axvline(6000,color = 'red',ls = '--')
plt.legend()

plt.xlim([5800,6100])
plt.yscale('log')
# Show the plot
plt.tight_layout()
plt.savefig('confidence_plots_grad.pdf')






