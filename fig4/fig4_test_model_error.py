# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 12:09:01 2025

@author: grijseelsd
"""
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
import os
import random
from scipy.stats import kstest

def measure_error_ind(binned_response, stim_len):
    data_folder = r'\\gpfs.corp.brain.mpg.de\bark\personal\grijseelsd\papers\nmr_antiphonal\paper_code\fig4\data'
    bins_all = np.load(os.path.join(data_folder,'data_periodic_noise_perc.npy'))
    if stim_len == 300:
        bins_all = bins_all[:,0,:]
    elif stim_len == 600:
        bins_all = bins_all[:,1,:]
    else:
        bins_all = bins_all[:,2,:]
    mean_bins = np.nanmean(bins_all/np.sum(bins_all,1)[:,None], 0)
    
    error_all = []
    for bins in bins_all:
        if np.sum(binned_response) == 0:
            error_all +=  [np.sum((bins)**2)]
        else:    
            perc_response = binned_response/np.sum(binned_response)
            error_all +=  [np.sum((perc_response-bins)**2)]
    
    return error_all
    
    

def inh_ramp_model(x, stim_len = 900):
    A_exc = x[0]
    inh_factor = x[1]
    r_mod = x[2]
    noise_inh_factor = x[3]
    threshold = 100
    
    data_folder = r'\\gpfs.corp.brain.mpg.de\bark\personal\grijseelsd\papers\nmr_antiphonal\paper_code\fig4\data'
    bins_all = np.load(os.path.join(data_folder,'data_periodic_noise_perc.npy'))
    if stim_len == 300:
        bins_all = bins_all[:,0,:]
    elif stim_len == 600:
        bins_all = bins_all[:,1,:]
    else:
        bins_all = bins_all[:,2,:]
    mean_bins = np.nanmean(bins_all/np.sum(bins_all,1)[:,None], 0)
    
    r_mod_inh = r_mod*inh_factor
    
    call_times = []
    stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
    n_rep = int(3*60*1000/(stim_len*2))
    for i in range(n_rep): # Run for 1000 repeats
  
        stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
        integrator = [0]
        exc_on = [0]
        A_prog = [A_exc]
        
        new_A_exc = A_exc
        for t in range(len(stim)):
            r = np.random.normal(0,1, 1)[0] # calculate the random modulation this timepoint
            exc_t1 = exc_on[t] + new_A_exc+r*r_mod# calculate the total excitation
            inh_t1 = -inh_factor*exc_on[t]+r*r_mod_inh
            integrate = exc_t1 - (noise_inh_factor*threshold*stim[t])
            if integrate>threshold: # if total reaches threshold, animal calls
                call_times += [t]
                break
            exc_on += [exc_t1]
            integrator.append(integrate)
            new_A_exc = (A_exc+inh_t1)
            A_prog += [new_A_exc]
      
    edges = np.linspace(0,1.8, 37)
    binned_response = np.histogram([call/1000 for call in call_times], edges)[0]
    
    if np.sum(binned_response) == 0:
        error =  np.sum((mean_bins)**2)
        perc_response = np.zeros(binned_response.shape)
    else:    
        perc_response = binned_response/np.sum(binned_response)
        error =  np.sum((perc_response-mean_bins)**2)
        
    error_p_ani = measure_error_ind(binned_response, stim_len)
    return error, perc_response, error_p_ani, [call/1000 for call in call_times]


def noise_inh_model(x, stim_len=900):
    A_exc = x[0]
    r_mod = x[1]
    noise_inh_factor = x[2]
    threshold = 100
    
    data_folder = r'\\gpfs.corp.brain.mpg.de\bark\personal\grijseelsd\papers\nmr_antiphonal\paper_code\fig4\data'
    bins_all = np.load(os.path.join(data_folder,'data_periodic_noise_perc.npy'))
    bins_all = bins_all[:,2,:]
    mean_bins = np.nanmean(bins_all/np.sum(bins_all,1)[:,None], 0)
    
    
    call_times = []
    stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
    n_rep = int(3*60*1000/(stim_len*2))
    for i in range(n_rep): # Run for 1000 repeats
  
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

    edges = np.linspace(0,1.8, 37)
    binned_response = np.histogram([call/1000 for call in call_times], edges)[0]
    
    if np.sum(binned_response) == 0:
        error =  np.sum((mean_bins)**2)
        perc_response = np.zeros(binned_response.shape)
    else:    
        perc_response = binned_response/np.sum(binned_response)
        error =  np.sum((perc_response-mean_bins)**2)
        
    error_p_ani = measure_error_ind(binned_response, stim_len)
    return error, perc_response, error_p_ani, [call/1000 for call in call_times]

def ramp_model(x, stim_len=900):
    A_on = x[0]
    A_off = x[1]
    r_mod_on = x[2]
    r_mod_off = x[3]

    data_folder = r'\\gpfs.corp.brain.mpg.de\bark\personal\grijseelsd\papers\nmr_antiphonal\paper_code\fig4\data'
    bins_all = np.load(os.path.join(data_folder,'data_periodic_noise_perc.npy'))
    bins_all = bins_all[:,2,:]
    mean_bins = np.nanmean(bins_all/np.sum(bins_all,1)[:,None], 0)
    
    data_folder =  r'\\gpfs.corp.brain.mpg.de\bark\personal\grijseelsd\papers\nmr_antiphonal\paper_code\fig3\data'
    # initiate paratemeter
    data = pd.read_csv(os.path.join(data_folder, 'strategy_per_animal.csv'))
    df_temp = data[np.logical_and(data['condition']==900, data['experiment']=='noise')]
    onset_perc = np.nanmean(df_temp[np.logical_or(data['epoch_type']==1, data['epoch_type']==3)]['percentage_all'])
    offset_perc = np.nanmean(df_temp[np.logical_or(data['epoch_type']==2, data['epoch_type']==3)]['percentage_all'])
    
    
    call_times = []
    stim = np.concat([np.ones((stim_len,)), np.zeros((stim_len, ))])
    n_rep = int(3*60*1000/(stim_len*2))
    for i in range(n_rep): # Run for 1000 repeats
  
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
                if integrate_on>100: # if total reaches threshold, animal calls
                    call_times += [t]
                    called_onset = True
                exc_on.append(exc_t1_on)
                integrator_on.append(integrate_on)
            if t>stim_len and initiate_offset and not called_offset: # offset ramping
                r = np.random.normal(0,1, 1) # calculate the random modulation this timepoint
                exc_t1_off = exc_off[t-stim_len-1]  +A_off+r*r_mod_off# calculate the total excitation
                integrate_off = exc_t1_off
                if integrate_off>100: # if total reaches threshold, animal calls
                    call_times += [t]
                    called_offset = True
                exc_off.append(exc_t1_off)
                integrator_off.append(integrate_off)
      
    edges = np.linspace(0,1.8, 37)
    binned_response = np.histogram([call/1000 for call in call_times], edges)[0]
    
    if np.sum(binned_response) == 0:
        error =  np.sum((mean_bins)**2)
        perc_response = np.zeros(binned_response.shape)
    else:    
        perc_response = binned_response/np.sum(binned_response)
        error =  np.sum((perc_response-mean_bins)**2)
        
    error_p_ani = measure_error_ind(binned_response, stim_len)
    return error, perc_response, error_p_ani, [call/1000 for call in call_times]


#######################################NOTE####################################
# Run fig4_fit_two_tier_model.py, fig4_fit_single_ramp and fig4_fit_single_ramp_with_feedback first to fit the models

###############################################################################

# Set random seed for reproducibility
random.seed(10)

# Data folders 
data_folder = os.path.join(os.path.dirname(os.getcwd()), 'fig2', 'data')
model_folder = os.path.join(os.getcwd(), 'data')
out_folder = os.path.join(os.getcwd(), 'data')


all_error = pd.DataFrame(columns=['data'])
all_model_response = pd.DataFrame(columns=['data'])
edges = np.linspace(0,1.8, 37)
edges_temp = (edges+np.nanmean(np.diff(edges))/2)[:-1]

# Load models
x_two_tier = np.load(os.path.join(data_folder, 'two_tier_result.npy'), allow_pickle=True)
x_noise = np.load(os.path.join(data_folder, 'single_ramp_result.npy'), allow_pickle=True)
x_feedback = np.load(os.path.join(data_folder, 'single_ramp_with_feedback_result.npy'), allow_pickle=True)

# Load call response curves outputted by fig2
all_calls_300ms = np.load(os.path.join(data_folder,'all_calls_300ms.npy'))
all_calls_600ms = np.load(os.path.join(data_folder,'all_calls_600ms.npy'))
all_calls_900ms = np.load(os.path.join(data_folder,'all_calls_900ms.npy'))


stim_lens = [300, 600, 900]
n_ani = 20
for stim in stim_lens:
    twotier_call_times = []
    feedback_call_times = []
    single_ramp_call_times = []
    for i in range(n_ani):
        error, response, error_p_ani, call_times  = inh_ramp_model(x_feedback, stim)
        error_df = pd.DataFrame(
                {'data': [error],
                 'error_p_ani': [np.nanmean(error_p_ani)],
                 'error_min': [np.nanmin(error_p_ani)],
                 'model': ['feedback'],
                 'rep': [i],
                'stim_len': [stim]})
        all_error = pd.concat([all_error, error_df])
        feedback_call_times+=call_times
        
        model_response = pd.DataFrame(
                {'data': response,
                 'time': edges_temp
            })
        model_response['model'] = 'feedback' 
        model_response['rep'] = i 
        model_response['stim_len'] = stim
        all_model_response = pd.concat([all_model_response, model_response])
        
        error, response, error_p_ani, call_times  = ramp_model(x_two_tier, stim)
        error_df = pd.DataFrame(
                {'data': [error],
                 'error_p_ani': [np.nanmean(error_p_ani)],
                 'error_min': [np.nanmin(error_p_ani)],
                 'model': ['ramp'],
                 'rep': [i],
                'stim_len': [stim]})
        all_error = pd.concat([all_error, error_df])
        model_response = pd.DataFrame(
                {'data': response,
                 'time': edges_temp,
            })
        model_response['model'] = 'ramp'   
        model_response['rep'] = i 
        model_response['stim_len'] = stim
        all_model_response = pd.concat([all_model_response, model_response])
        twotier_call_times+=call_times
        
        error, response, error_p_ani, call_times  = noise_inh_model(x_noise, stim)
        error_df = pd.DataFrame(
                {'data': [error],
                 'error_p_ani': [np.nanmean(error_p_ani)],
                 'error_min': [np.nanmin(error_p_ani)],
                 'model': ['noise'],
                 'rep': [i],
                'stim_len': [stim]})
        all_error = pd.concat([all_error, error_df])
        model_response = pd.DataFrame(
                {'data': response,
                 'time': edges_temp
            })
        model_response['model'] = 'noise'  
        model_response['rep'] = i 
        model_response['stim_len'] = stim
        all_model_response = pd.concat([all_model_response, model_response])
        single_ramp_call_times+=call_times
    if stim==300:
        all_calls = all_calls_300ms
    elif stim==600:
        all_calls = all_calls_600ms
    else:
        all_calls = all_calls_900ms
    print(stim)
    
    print('two-tier')
    print(kstest(twotier_call_times, all_calls))
    
    print('single_ramp')
    print(kstest(single_ramp_call_times, all_calls))
    
    print('feedback')
    print(kstest(feedback_call_times, all_calls))
    

all_error.to_csv(os.path.join(out_folder,'model_errors.csv'), index=False)  
all_model_response.to_csv(os.path.join(out_folder,'model_responses.csv'), index=False)  
