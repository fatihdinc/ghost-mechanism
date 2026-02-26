#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 19:58:18 2026

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
f = np.load('long_training_fp_test.npz')

loss=f['original_loss_history']
kappa_dkappa= f['original_dkappa_vs_kappa_history']
grad=f['grad_history']
kappa_initial=f['kappa_all']

plt.semilogy(loss)
plt.semilogy(grad,alpha = 0.5)
plt.savefig('loss_grad.pdf')
plt.show()

#%%

def count_flips(array):
    flips = 0
    for i in range(1, len(array)):
        if array[i] != array[i-1]:
            flips += 1
    return flips

def find_num_fp(dk):
    dk = dk > 0
    return count_flips(dk)



dkappa = kappa_dkappa[:,1,:]

fps = np.zeros(dkappa.shape[0])


pbar = tqdm(range(dkappa.shape[0]))
for i in pbar:
    fps[i] = np.max(find_num_fp(dkappa[i, :]))
    pbar.set_postfix(max_fps=np.max(fps))

print(np.max(fps))


