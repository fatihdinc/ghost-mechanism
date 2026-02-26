#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:44:39 2024

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt



f = np.load('loss_save_dms.npz')
loss_all = f['loss_all']
grad_all = f['grad_all']
weight_all = f['weight_all']

lr_all = f['lr_all']

plt.figure(figsize=(7,2))

tt = 0

data = np.atleast_2d(lr_all) * np.ones([20,1])

temp = np.mean(loss_all[:,:,tt:],2)

w = 0.1
width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)


x =  np.atleast_2d(lr_all) * np.ones([20,1])  + np.random.normal(size=data.shape)*lr_all/20


plt.scatter(x,temp,s=3)

plt.boxplot(temp, positions = lr_all, widths= width(lr_all,w),sym='')




temp = np.mean(grad_all[:,:,tt:],2)

w = 0.1
width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)


plt.scatter(x,temp,s=3)

plt.boxplot(temp, positions = lr_all, widths= width(lr_all,w),sym='')


temp = np.mean(weight_all[:,:,tt:],2)

w = 0.1
width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)


plt.scatter(x,temp,s=10)

plt.boxplot(temp, positions = lr_all, widths= width(lr_all,w),sym='')
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-6,1e0])

plt.savefig('lr_summary.pdf')

plt.show()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
lr = 15

pick = 12
    
plt.loglog(loss_all[pick,lr,:],color = colors[0])
plt.loglog(grad_all[pick,lr,:],color = colors[1])
plt.loglog(weight_all[pick,lr,:],color = colors[2])
plt.xlim([10,3e4])
plt.savefig('case1.pdf')
plt.show()

pick = 0
plt.loglog(loss_all[pick,lr,:],color = colors[0])
plt.loglog(grad_all[pick,lr,:],color = colors[1])
plt.loglog(weight_all[pick,lr,:],color = colors[2])
plt.xlim([10,3e4])
plt.savefig('case2.pdf')
