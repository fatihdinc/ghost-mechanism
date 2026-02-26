#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:17:40 2024

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt


f = np.load('long_training.npz')

loss=f['original_loss_history']
kappa_dkappa= f['original_dkappa_vs_kappa_history']
grad=f['grad_history']
kappa_initial=f['kappa_all']



plt.semilogy(np.arange(1,loss.shape[0]+1),loss/np.max(loss))
plt.semilogy(np.arange(1,loss.shape[0]+1),grad/np.max(grad),alpha = 0.5)
plt.ylim([1e-6,5])
plt.savefig('Fig3a.pdf')
plt.show()

#%%

y = kappa_dkappa[0]
plt.plot(y[0][200:800],y[1][200:800],label = '1')


y = kappa_dkappa[200]
plt.plot(y[0][200:800],y[1][200:800],label = '200,001')

y = kappa_dkappa[-1]
plt.plot(y[0][200:800],y[1][200:800],label = '499,001')

plt.scatter(kappa_initial,np.zeros(kappa_initial.shape),50,'red','.')
plt.axhline(0,color = 'black',ls = '--')
plt.axvline(1,color = 'orange',ls = '--')
plt.xlabel('kappa')
plt.ylabel('dkappa/dt')
plt.legend()
plt.savefig('Fig3b.pdf')

plt.show()


