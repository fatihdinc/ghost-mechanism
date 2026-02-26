#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 12:53:43 2024

@author: dinc
"""

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-1, 1, 100)

# Plot with positive offset
plt.plot(x, x**2 + 0.5)
plt.axhline(0, color='black', linewidth=0.5)  # Add horizontal line at y = 0
plt.ylim(-1, 2)  # Set y-limits to ensure y=0 is visible
plt.savefig('pos.pdf')
plt.show()

# Plot with negative offset
plt.plot(x, x**2 - 0.5)
plt.axhline(0, color='black', linewidth=0.5)  # Add horizontal line at y = 0
plt.ylim(-1, 2)  # Set y-limits to ensure y=0 is visible
plt.savefig('neg.pdf')
plt.show()