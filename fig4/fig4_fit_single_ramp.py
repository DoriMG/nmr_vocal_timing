# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:35:41 2025

@author: door1
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def model_func(x, data_folder):
    A_exc = x[0]
    r_mod = x[1]
    noise_inh_factor = x[2]
    stim_len = 900
    threshold = 100
    
    bins_all = np.load(os.path.join(data_folder,'data_periodic_noise_perc.npy'))
    bins_all = bins_all[:,2,:]
    mean_bins = np.nanmean(bins_all/np.sum(bins_all,1)[:,None], 0)
    
    
    call_times = []
    stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
    for i in range(1000): # Run for 1000 repeats
  
        stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
        integrator = [0]
        exc_on = [0]

        for t in range(len(stim)):
            r = np.random.normal(0,1, 1)[0] # calculate the random modulation this timepoint
            exc_t1 = exc_on[t] + A_exc+r*r_mod# calculate the total excitation
            integrate = exc_t1 - (noise_inh_factor*threshold*stim[t])
            if integrate>threshold: # if total reaches threshold, animal calls
                call_times += [t]
                break
            exc_on += [exc_t1]
            integrator.append(integrate)
      
    plt.hist(call_times,100)
    edges = np.linspace(0,1.8, 37)
    binned_response = np.histogram([call/1000 for call in call_times], edges)[0]
    
    if np.sum(binned_response) == 0:
        error =  np.sum((mean_bins)**2)
    else:    
        perc_response = binned_response/np.sum(binned_response)
        error =  np.sum((perc_response-mean_bins)**2)
        
    return error

# data folders
data_folder = os.path.join(os.path.dirname(os.getcwd()), 'fig2', 'data')
out_folder = os.path.join(os.getcwd(), 'data')

# Load original data from fig 2
bins_all = np.load(os.path.join(data_folder,'data_periodic_noise_perc.npy'))
bins_all = bins_all[:,2,:]
mean_bins = np.nanmean(bins_all/np.sum(bins_all,1)[:,None], 0)


all_model_response = pd.DataFrame(columns=['data','time', 'mean_delay'])
# Initial search space for variables
A_exc_vary = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
r_mod_vary = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 4, 6, 8]
noise_inh_factor_vary = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# Run through initial parameter space
stim_len = 900
all_error = pd.DataFrame(columns=['data'])
for A_exc_idx, A_exc  in enumerate(A_exc_vary):
        for r_mod_idx, r_mod  in enumerate(r_mod_vary):
            for noise_inh_factor_idx, noise_inh_factor  in enumerate(noise_inh_factor_vary):
                print('A_exc: ' + str(A_exc_idx)+'; r_mod'+ str(r_mod_idx)+'; noise_inh_factor'+ str(noise_inh_factor_idx))
                x = [A_exc, r_mod, noise_inh_factor]
                error = model_func(x, data_folder)
                error_df = pd.DataFrame(
                        {'data': [error],
                        'A_exc': [A_exc_idx+6],
            
                        'r_mod': [r_mod_idx],
                        'noise_inh_factor': [noise_inh_factor_idx]})
                
                all_error = pd.concat([all_error, error_df])
all_error.to_csv(os.path.join(out_folder,'single_ramp_grid_search.csv'), index=False)  

# Calculate optimal from grid search
min_row = all_error[all_error['data']==all_error['data'].min()]
A_exc = A_exc_vary[int(min_row['A_exc'].values[0])]
r_mod = r_mod_vary[int(min_row['r_mod'].values[0])]
noise_inh_factor = noise_inh_factor_vary[int(min_row['noise_inh_factor'].values[0])]

# Minimize further using scipy
x0 = [A_exc, r_mod, noise_inh_factor]
bnds = ((0, None), (0, None), (0,None))
res = minimize(model_func, x0, method='Powell', bounds=bnds, options={'disp': True})
x = res.x
np.save(os.path.join(out_folder,'single_ramp_result.npy'), x, allow_pickle=True)


# Fitted parameters
A_exc = x[0]
r_mod = x[1]
noise_inh_factor = x[2]
threshold = 100

# Run the model
all_model_response = pd.DataFrame(columns=['data','time', 'stim_len'])
for stim_len in [300, 600, 900]:
    call_times = []
    stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
    for i in range(5000): # Run for 1000 repeats
      
        stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
        integrator = [0]
        exc_on = [0]

        for t in range(len(stim)):
            r = np.random.normal(0,1, 1)[0] # calculate the random modulation this timepoint
            exc_t1 = exc_on[t] + A_exc+r*r_mod# calculate the total excitation
            integrate = exc_t1 - (noise_inh_factor*threshold*stim[t])
            if integrate>threshold: # if total reaches threshold, animal calls
                call_times += [t]
                break
            exc_on += [exc_t1]
            integrator.append(integrate)
      
    plt.hist(call_times,100)
    edges = np.linspace(0,1.8, 37)
    binned_response = np.histogram([call/1000 for call in call_times], edges)[0]
    
    if np.sum(binned_response) == 0:
        error =  np.sum((mean_bins)**2)
    else:    
        perc_response = binned_response/np.sum(binned_response)
        error =  np.sum((perc_response-mean_bins)**2)
        
    edges_temp = (edges+np.nanmean(np.diff(edges))/2)[:-1]
    model_response = pd.DataFrame(
            {'data': binned_response/np.sum(binned_response),
         'time': edges_temp,
         'stim_len':stim_len
        })
    all_model_response = pd.concat([all_model_response, model_response])

all_model_response.to_csv(os.path.join(out_folder,'fitted_model_single_ramp.csv'), index=False)  
