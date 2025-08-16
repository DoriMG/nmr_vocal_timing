# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 19:35:41 2025

@author: door1
"""
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def model_func(x, data_folder, strategy_folder):
    A_on = x[0]
    A_off = x[1]
    r_mod_on = x[2]
    r_mod_off = x[3]
    stim_len = 900
    threshold = 100
    
    # Load original data
    bins_all = np.load(os.path.join(data_folder,'data_periodic_noise_perc.npy'))
    bins_all = bins_all[:,2,:]
    mean_bins = np.nanmean(bins_all/np.sum(bins_all,1)[:,None], 0)

    # Load animal strategies
    data = pd.read_csv(os.path.join(strategy_folder, 'strategy_per_animal.csv'))
    df_temp = data[np.logical_and(data['condition']==900, data['experiment']=='noise')]
    onset_perc = np.nanmean(df_temp[np.logical_or(data['epoch_type']==1, data['epoch_type']==3)]['percentage_all'])
    offset_perc = np.nanmean(df_temp[np.logical_or(data['epoch_type']==2, data['epoch_type']==3)]['percentage_all'])
    
    
    call_times = []
    stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
    for i in range(1000): # Run for 1000 repeats
  
        stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
        integrator_on = [0]
        exc_on = [0]
        
        integrator_off = [0]
        exc_off = [0]
        called_onset = False
        called_offset = False
        
        initiate_onset = random.random()<onset_perc
        initiate_offset = random.random()<offset_perc
        for t in range(len(stim)): # for each time point (milisecond)
            if initiate_onset and not called_onset:#start onset ramping            
                r = np.random.normal(0,1, 1) # calculate the random modulation this timepoint
                exc_t1_on = exc_on[t]  +A_on+r*r_mod_on# calculate the total excitation
                integrate_on = exc_t1_on
                if integrate_on>threshold: # if total reaches threshold, animal calls
                    call_times += [t]
                    called_onset = True
                exc_on.append(exc_t1_on)
                integrator_on.append(integrate_on)
            if t>stim_len and initiate_offset and not called_offset: # offset ramping
                r = np.random.normal(0,1, 1) # calculate the random modulation this timepoint
                exc_t1_off = exc_off[t-stim_len-1]  +A_off+r*r_mod_off# calculate the total excitation
                integrate_off = exc_t1_off
                if integrate_off>threshold: # if total reaches threshold, animal calls
                    call_times += [t]
                    called_offset = True
                exc_off.append(exc_t1_off)
                integrator_off.append(integrate_off)
      
    plt.hist(call_times,100)
    edges = np.linspace(0,1.8, 37)
    binned_response = np.histogram([call/1000 for call in call_times], edges)[0]
    
    if np.sum(binned_response) == 0:
        error =  np.sum((mean_bins)**2)
    else:    
        perc_response = binned_response/np.sum(binned_response)
        error =  np.sum((perc_response-mean_bins)**2)
        
    return error


data_folder = os.path.join(os.path.dirname(os.getcwd()), 'fig2', 'data')
strategy_folder = os.path.join(os.path.dirname(os.getcwd()), 'fig3', 'data')
out_folder = os.path.join(os.getcwd(), 'data')

# Load original data from fig 2
bins_all = np.load(os.path.join(data_folder,'data_periodic_noise_perc.npy'))
bins_all = bins_all[:,2,:]
mean_bins = np.nanmean(bins_all/np.sum(bins_all,1)[:,None], 0)


all_model_response = pd.DataFrame(columns=['data','time', 'mean_delay'])

# Initial search space for variables
A_on_vary = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
A_off_vary = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
r_mod_on_vary = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 4, 6, 8]
r_mod_off_vary = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 4, 6, 8]

# Run through initial parameter space
stim_len = 900
threshold = 100
all_error = pd.DataFrame(columns=['data'])
for A_on_idx, A_on  in enumerate(A_on_vary):
    for A_off_idx, A_off  in enumerate(A_off_vary):
        for r_mod_on_idx, r_mod_on  in enumerate(r_mod_on_vary):
            for r_mod_off_idx, r_mod_off  in enumerate(r_mod_off_vary):
                print('A_on: ' + str(A_on_idx)+'; A_off'+ str(A_off_idx)+'; r_mod_on'+ str(r_mod_on_idx)+'; r_mod_off'+ str(r_mod_off_idx))
                x = [A_on, A_off, r_mod_on, r_mod_off]
                error = model_func(x, data_folder, strategy_folder)
                error_df = pd.DataFrame(
                        {'data': [error],
                        'A_on': [A_on_idx],
                        'A_off': [A_off_idx],
                        'r_mod_on': [r_mod_on_idx],
                        'r_mod_off': [r_mod_off_idx]})
                
                all_error = pd.concat([all_error, error_df])

all_error.to_csv(os.path.join(out_folder,'two_tier_grid_search.csv'), index=False)  

# Calculate optimal from grid search
min_row = all_error[all_error['data']==all_error['data'].min()]
A_on = A_on_vary[int(min_row['A_on'].values[0])]
A_off = A_off_vary[int(min_row['A_off'].values[0])]
r_mod_on = r_mod_on_vary[int(min_row['r_mod_on'].values[0])]
r_mod_off = r_mod_off_vary[int(min_row['r_mod_off'].values[0])]

# Minimize further using scipy
x0 = [A_on, A_off, r_mod_on, r_mod_off]
res = minimize(model_func, x0, method='Powell',  options={'disp': True})
x = res.x
np.save(os.path.join(out_folder,'two_tier_result.npy'), x, allow_pickle=True)


# Fitted parameters
A_on = x[0]
A_off = x[1]
r_mod_on = x[2]
r_mod_off = x[3]

# Find the offset/onset parameters for animals
data = pd.read_csv(os.path.join(strategy_folder, 'strategy_per_animal.csv'))
df_temp = data[np.logical_and(data['condition']==900, data['experiment']=='noise')]
onset_perc = np.nanmean(df_temp[np.logical_or(data['epoch_type']==1, data['epoch_type']==3)]['percentage_all'])
offset_perc = np.nanmean(df_temp[np.logical_or(data['epoch_type']==2, data['epoch_type']==3)]['percentage_all'])

all_model_response = pd.DataFrame(columns=['data','time', 'stim_len'])
for stim_len in [300, 600, 900]:
    call_times = []
    stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
    for i in range(5000): # Run for 5000 repeats
        integrator_on = [0]
        exc_on = [0]
        
        integrator_off = [0]
        exc_off = [0]
        called_onset = False
        called_offset = False
        
        initiate_onset = random.random()<onset_perc
        initiate_offset = random.random()<offset_perc
        for t in range(len(stim)): # for each time point (milisecond)
            if initiate_onset and not called_onset:#start onset ramping            
                r = np.random.normal(0,1, 1) # calculate the random modulation this timepoint
                exc_t1_on = exc_on[t]  +A_on+r*r_mod_on# calculate the total excitation
                integrate_on = exc_t1_on
                if integrate_on>threshold: # if total reaches threshold, animal calls
                    call_times += [t]
                    called_onset = True
                exc_on.append(exc_t1_on)
                integrator_on.append(integrate_on)
            if t>stim_len and initiate_offset and not called_offset: # offset ramping
                r = np.random.normal(0,1, 1) # calculate the random modulation this timepoint
                exc_t1_off = exc_off[t-stim_len-1]  +A_off+r*r_mod_off# calculate the total excitation
                integrate_off = exc_t1_off
                if integrate_off>threshold: # if total reaches threshold, animal calls
                    call_times += [t]
                    called_offset = True
                exc_off.append(exc_t1_off)
                integrator_on.append(integrate_off)
             
    plt.hist(call_times,100)
    
    edges = np.linspace(0,1.8, 37)
    binned_response = np.histogram([call/1000 for call in call_times], edges)[0]
    edges_temp = (edges+np.nanmean(np.diff(edges))/2)[:-1]
    
    model_response = pd.DataFrame(
            {'data': binned_response/np.sum(binned_response),
         'time': edges_temp
        })
    model_response['stim_len'] = stim_len
    all_model_response = pd.concat([all_model_response, model_response])
    
all_model_response.to_csv(os.path.join(out_folder,'fitted_two_tier_model.csv'), index=False) 

