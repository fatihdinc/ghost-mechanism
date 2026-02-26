#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:17:40 2024

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt



f = np.load('long_training_large_lr.npz')

loss=f['original_loss_history']
kappa_dkappa= f['original_dkappa_vs_kappa_history']
grad=f['grad_history']
kappa_initial=f['kappa_all']


plt.semilogy(np.arange(1,loss.shape[0]+1),loss/np.max(loss))
plt.semilogy(np.arange(1,loss.shape[0]+1),grad/np.max(grad))
plt.ylim([1e-6,5])
plt.savefig('Fig3c.pdf')
plt.show()

plt.semilogy(np.arange(1,loss.shape[0]+1),loss/np.max(loss))
plt.xlim([5000,5200])
plt.axvline(5156)
plt.axvline(5163)
plt.axvline(5166)
plt.savefig('Fig3c-inset.pdf')
plt.show()

#%%



y = kappa_dkappa[5155]
plt.plot(y[0][200:800],y[1][200:800],label = '5156')


y = kappa_dkappa[5162]
plt.plot(y[0][200:800],y[1][200:800],label = '5163')

y = kappa_dkappa[5165]
plt.plot(y[0][200:800],y[1][200:800],label = '5166')
plt.ylim([-1,3])

plt.scatter(kappa_initial,np.zeros(kappa_initial.shape),50,'red','.')
plt.axhline(0,color = 'black',ls = '--')
plt.axvline(1,color = 'orange',ls = '--')
plt.xlabel('kappa')
plt.ylabel('dkappa/dt')
plt.legend()
plt.savefig('Fig3d.pdf')




