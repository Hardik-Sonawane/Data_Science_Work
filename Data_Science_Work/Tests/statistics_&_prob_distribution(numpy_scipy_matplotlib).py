# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:20:43 2024

@author: scs
"""

############ Test-8 ###################3

'''
Que.1 Given a dataset of integers or floating-point numbers, calculate the 
following descriptive statistics:
 Mean
 Median
 Mode
 Variance
 Standard Deviation
Sample Dataset: [20, 40, 40, 40, 30, 50, 60]
'''

import numpy as np
from scipy import stats
sample_data = [20, 40, 40, 40, 30, 50, 60]

mean = np.mean(sample_data)
median = np.median(sample_data)
mode = stats.mode(sample_data)
variance = np.var(sample_data)
std_devn = np.std(sample_data)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_devn}")
#############################3
'''
Ques.2    Generate a dataset of 1,000 random values generated from a lognormal 
distribution with a mean of 0 and a standard deviation of 1 in the log-space, 
perform the following tasks:
 Plot the histogram of the dataset.
 Calculate the mean and median of the dataset.
 Fit a lognormal distribution to the data and overlay the probability density 
function (PDF) on the histogram.
 '''
 
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

data = np.random.lognormal(mean=0, sigma=1, size=1000)

plt.hist(data, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black')

a = np.mean(data)
b = np.median(data)
print(f"Mean: {a:.2f}, Median: {b:.2f}")

shape, loc, scale = lognorm.fit(data, floc=0)
x = np.linspace(min(data), max(data), 1000)
plt.plot(x, lognorm.pdf(x, shape, loc, scale), 'r-', lw=2)


plt.title('Histogram and Fitted Lognormal PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

#####################''
'''
Que 3. Generate 1,000 random values following a logarithmic distribution with 
a probability parameter p = 0.3. Perform the following tasks:
 Plot the histogram of the dataset.
 Calculate the mean of the dataset.
 Overlay the probability mass function (PMF) of the logarithmic 
distribution on the histogram.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logser

p = 0.3
a = logser.rvs(p, size=1000)

plt.hist(a, bins=range(1, max(a) + 1), density=True, alpha=0.6, color='blue', edgecolor='black')

n = np.mean(a)
print(f"Mean: {n:.2f}")

x = np.arange(1, max(a) + 1)
plt.plot(x, logser.pmf(x, p), 'r-', lw=2)

plt.title('H and L PMF')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.show()

#######################################
