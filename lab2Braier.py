#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:26:27 2022
Modified on Fri Mar 11 15:26:27 2022
@author: Your name

Description
------------
"""

# Part 1 - Histogram the data

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(1,figsize=(6,6))
fig.clf()
axes = [fig.add_subplot(321),\
        fig.add_subplot(322),\
        fig.add_subplot(323),\
        fig.add_subplot(324),\
        fig.add_subplot(325),\
        fig.add_subplot(326)]

# You can use axes[0], axes[1], ....  axes[5] to make the six histograms.

# Your code goes here

beta_deg = np.loadtxt('refractiondata.txt', skiprows=3) # row n is 10(n+1) degrees

alpha_deg = np.linspace(10,60,6)

for i in range(len(axes)):
    axes[i].hist(beta_deg[i], bins=15, range=(-10,50))
    axes[i].set_xlabel('beta (deg.)')
    axes[i].set_title(f'alpha = {int(alpha_deg[i])} deg.')
    axes[i].set_xticks(np.linspace(-10,50,7))

fig.tight_layout()


#%%

# Part 2 - Table of measurements

# Your code goes here
na = len(beta_deg)

beta_rad = beta_deg*np.pi/180

print(' alpha\t| sin(alpha)\t| beta\t\t| sin(beta)\t| sigma(sin(beta))')
print(65*'-')

sin_beta = np.zeros(na)
error_beta = np.zeros(na)

for i in range(na):
    a = alpha_deg[i] # one value
    b_data = beta_deg[i] # 16 values
    
    sin_a = np.sin(a*np.pi/180)
    b = np.mean(b_data) # average b
    sin_b = np.sin(b*np.pi/180)
    # sigma_b_mean = np.std(b_data)/np.sqrt(len(b_data))
    
    sin_b_data = np.sin(b_data*np.pi/180)
    sigma_sin_b = np.std(sin_b_data)/np.sqrt(len(b_data))
   
    # assigning variables to the outside
    sin_beta[i] = sin_b
    error_beta[i] = sigma_sin_b
    print(f' {a:.0f}\t\t| {sin_a:.3f}\t\t| {b:6.3f}\t\t| {sin_b:.3f}\t\t| {sigma_sin_b:.3f}')

#%%

# Part 3 - Snells law plot and fit

fig = plt.figure(2,figsize=(6,6))
fig.clf()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# You can use ax1 and ax2 for the Snell's law plot and the chi squared plot.

# Your code goes here
from scipy.optimize import curve_fit
import scipy.stats as st

sin_alpha = np.sin(alpha_deg*np.pi/180)

ax1.errorbar(sin_alpha, sin_beta, error_beta, fmt='ok', label='data', markersize=3)
ax1.set_xlabel('sin($\\alpha$)', fontsize=14)
ax1.set_ylabel('sin($\\beta$)', fontsize=14)
ax1.set_title('sin($\\alpha$) vs. sin($\\beta$)', fontsize = 16)

def snell(sin_a, n):
    sin_b = sin_a/n
    return sin_b

p0 = (1,)
params, covar = curve_fit(snell, sin_alpha, sin_beta, p0, sigma=error_beta, absolute_sigma=True)

nglass = params[0]
nerr = np.sqrt(covar[0][0])

xx_fit = np.linspace(0,1,10)
yy_fit = 1/nglass * xx_fit # could also use the snell function here
yy_fit_up = 1/(nglass+nerr) * xx_fit
yy_fit_down = 1/(nglass-nerr) * xx_fit

ax1.plot(xx_fit, yy_fit, '-r', label = 'fit')
ax1.legend()

plt.tight_layout()

print()
print(f'n = {nglass:.2f} +/- {nerr:.2f}')

# calculate chisq
chisq = ((sin_beta-snell(sin_alpha,nglass))/error_beta)**2
pval= st.chi2.sf(np.sum(chisq), 5) # 6 data points, one fitting parameter
print()
print(f'chisq={np.sum(chisq):.4f}, dof=5, pval={pval:.4f}')



#%%
# Part 4 - Chi squared plot

# Your code goes here


# only want one data point per alpha, not 16

def chi(nglass):
    data = sin_beta # 6 sin beta averages
    model = snell(sin_alpha, nglass) # 6 sin beta models
    unc = error_beta # 6 beta uncertainties
    chisq = ((data-model)/unc)**2 # 6 chisq values
    return np.sum(chisq)

nn = np.linspace(1.42,1.55,51)
chi_array = np.zeros(len(nn))

for i in range(len(nn)):
    chi_array[i] = chi(nn[i])

ax2.plot(nn, chi_array, color='r')
ax2.set_xlabel('index of refraction', fontsize=14)
ax2.set_ylabel('$\\chi^2$', fontsize=14)
ax2.set_title('$\\chi^2$ for different indices of refraction', fontsize=16)

# adding horizontal and vertical lines

chimin = min(chi_array)
plt.hlines(chimin, 1.45,1.525, color='k',ls='--')
plt.hlines(chimin+1, 1.45,1.525, color='k',ls='--')

plt.vlines(nglass, 0,10, color='k',ls='--')
plt.vlines(nglass+nerr,0,10,color='k',ls='--')
plt.vlines(nglass-nerr,0,10,color='k',ls='--')

fig.tight_layout()




######################################################


