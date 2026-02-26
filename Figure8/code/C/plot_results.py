#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 23:40:57 2025

@author: dinc
"""


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

from datetime import datetime
c_all = [0.1,0.5,1,3,np.inf]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors.insert(5,'black')


f =  np.load('intervention_results.npz')


acc=f['acc_all']
grad=f['grad_all']

#%%

met = acc
epochs = np.linspace(1,20000,100)

for i in range(5):
    
    plt.errorbar(epochs,np.mean(met[:,i,::200],0),np.std(met[:,i,::200],0)/np.sqrt(100),color =colors[i],label = c_all[i])

plt.ylim([0,1])
plt.legend()
plt.savefig('accuracies.pdf')
plt.show()

