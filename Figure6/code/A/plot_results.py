#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:44:39 2024

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt



f = np.load('loss_save_full.npz')
loss_all = f['loss_all']
grad_all = f['grad_all']

lr_all = np.logspace(-4,1,30)


cmap = plt.cm.autumn   # you can swap this for plasma, inferno, magma, cividis
colors = cmap(np.linspace(0.1, 0.9, 6))




plt.errorbar(lr_all,loss_all.mean(0)[:,10000-1],
             loss_all.std(0)[:,10000-1]/np.sqrt(loss_all.shape[0]), color = colors[0],
             label = '10000')
plt.errorbar(lr_all,loss_all.mean(0)[:,20000-1],
             loss_all.std(0)[:,20000-1]/np.sqrt(loss_all.shape[0]), color = colors[1],
             label = '20000')
plt.errorbar(lr_all,loss_all.mean(0)[:,30000-1],
             loss_all.std(0)[:,30000-1]/np.sqrt(loss_all.shape[0]),color = colors[2],
             label = '30000')
plt.errorbar(lr_all,loss_all.mean(0)[:,50000-1],
             loss_all.std(0)[:,50000-1]/np.sqrt(loss_all.shape[0]),color = colors[3],
             label = '50000')
plt.errorbar(lr_all,loss_all.mean(0)[:,100000-1],
             loss_all.std(0)[:,100000-1]/np.sqrt(loss_all.shape[0]),color = colors[4],
             label = '100000')
plt.errorbar(lr_all,loss_all.mean(0)[:,200000-1],
             loss_all.std(0)[:,200000-1]/np.sqrt(loss_all.shape[0]), color = colors[5],
             label = '200000')

plt.yscale('log')
plt.xscale('log')
plt.legend(title = 'Epoch')
plt.xlabel('Learning rate')
plt.ylabel('Loss value')
plt.ylim([1e-7,3])
plt.savefig('rnn_lr_full.pdf')
plt.show()

#%%




f = np.load('loss_save.npz')
loss_all = f['loss_all']
grad_all = f['grad_all']

lr_all = np.logspace(-4,-1,30)

plt.errorbar(lr_all,loss_all.mean(0)[:,10000-1],
             loss_all.std(0)[:,10000-1]/np.sqrt(loss_all.shape[0]), color = colors[0],
             label = '10000')
plt.errorbar(lr_all,loss_all.mean(0)[:,20000-1],
             loss_all.std(0)[:,20000-1]/np.sqrt(loss_all.shape[0]), color = colors[1],
             label = '20000')
plt.errorbar(lr_all,loss_all.mean(0)[:,30000-1],
             loss_all.std(0)[:,30000-1]/np.sqrt(loss_all.shape[0]),color = colors[2],
             label = '30000')
plt.errorbar(lr_all,loss_all.mean(0)[:,50000-1],
             loss_all.std(0)[:,50000-1]/np.sqrt(loss_all.shape[0]),color = colors[3],
             label = '50000')
plt.errorbar(lr_all,loss_all.mean(0)[:,100000-1],
             loss_all.std(0)[:,100000-1]/np.sqrt(loss_all.shape[0]),color = colors[4],
             label = '100000')
plt.errorbar(lr_all,loss_all.mean(0)[:,200000-1],
             loss_all.std(0)[:,200000-1]/np.sqrt(loss_all.shape[0]), color = colors[5],
             label = '200000')

plt.yscale('log')
plt.xscale('log')
plt.legend(title = 'Epoch')
plt.xlabel('Learning rate')
plt.ylabel('Loss value')
plt.ylim([1e-7,3])
plt.savefig('rnn_lr_rank-one.pdf')
plt.show()