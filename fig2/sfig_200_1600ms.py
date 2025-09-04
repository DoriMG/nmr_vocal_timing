# -*- coding: utf-8 -*-
"""
@author: Dori M. Grijseels
"""

import pandas as pd
import numpy as np
import os
import librosa

# Custom functions import
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'util'))
from preprocessing import find_all_text_files, load_calls, convert_to_R

def call_per_stim(start_time, stim_times, burst_length, df):

    calls_during_noise = []
    calls_per_stim = []
    for stim in stim_times:
        current_start = stim + start_time
        current_end = current_start + (burst_length*2)
        df_temp = df[(df['end']<current_end) & (df['start']>current_start)]
        calls_during_noise += list(df_temp['start'] - current_start)
        if len(df_temp)>0:
            calls_per_stim +=[1]
        else:
            calls_per_stim += [0]
        current_start = current_end

    return calls_during_noise, calls_per_stim


def get_start_times(conditions, stim_folder):
    start_time = {}
    for j,c in enumerate(conditions):
        
        audio = os.path.join(stim_folder, c.replace('_', '') + '.wav')
        y, sr = librosa.load(audio, sr=None)
        frame_length = int(sr * 0.001)
        hop_length = frame_length

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        silence_threshold = 0.0001

        noise = np.insert((rms>silence_threshold).astype(float),0,0)
        start_stim = np.where(np.diff(noise)==1)[0]/(sr/frame_length)
        
        start_time[c] = start_stim
    return start_time


# Data folders 
data_folder = os.path.join(os.path.dirname(os.getcwd()), 'data', 'noise_long_short')
out_folder = os.path.join(os.getcwd(), 'data')
stim_folder = os.path.join(os.path.dirname(os.getcwd()), 'metadata', 'stimuli')

# Load data files
all_files = find_all_text_files([data_folder])


# Settings
condition_segments = {}
conditions = ['1600noise', '200noise']
period_time = 0.9
session_lengths = {} 
edges = np.linspace(0,1.8, 37)
min_tot_calls = 5

# Control for inaccuracies in stimulus creation
stim_time = get_start_times(conditions, stim_folder)


# Arrays to save out
bins_all =  np.full([len(all_files),len(conditions), 36], np.nan)
bins_all_perc =  np.full([len(all_files),len(conditions), 36], np.nan)
hist_data = pd.DataFrame(columns=['data','condition', 'animal'])
num_calls = pd.DataFrame(columns=['data', 'condition', 'animal'])
num_calls_p_epoch = pd.DataFrame(columns=['data', 'condition', 'animal'])

all_calls_200ms = []
all_calls_1600ms = []

# Histogram per epoch
for i, file in enumerate(all_files):
    df = load_calls(file)
    filename = os.path.basename(file)
    animal_id = filename.split('_')[0]
    for j,c in enumerate(conditions):
        start_time = df.loc[df['label'] == 'start_'+c, 'start'].values[0]
        end_time = df.loc[df['label'] == 'end_'+c, 'start'].values[0]
        
        # get the total number of calls in this period, check if more than a ceratin minimum
        tot_calls = sum((df['start']>start_time) & (df['end']<end_time))
        df_temp = pd.DataFrame.from_dict({'data': [tot_calls], 'condition': [c], 'animal': [animal_id]})
        num_calls = pd.concat([num_calls, df_temp])
        
        
        calls, epoch_with_call = call_per_stim(start_time, stim_time[c],period_time, df)
        if tot_calls>min_tot_calls:
            bins_all[i,j,:] = np.histogram(calls, edges)[0]
            temp_hist = np.histogram(calls, edges)[0]
            bins_all_perc[i,j,:] =temp_hist/np.nansum(temp_hist)
        if tot_calls>0:
            df_temp = pd.DataFrame.from_dict({'data': calls})
            df_temp['condition'] = c
            df_temp['animal'] = animal_id
            hist_data = pd.concat([hist_data, df_temp])
            
            df_temp = pd.DataFrame.from_dict({'data': [np.nanmean(epoch_with_call)], 'condition': [c], 'animal': [animal_id]})
        else:
            df_temp = pd.DataFrame.from_dict({'data': [0], 'condition': [c],'animal': [animal_id]})
        
        num_calls_p_epoch = pd.concat([num_calls_p_epoch, df_temp])
        
        if j==0:
            all_calls_200ms+=calls
        elif j==1:
            all_calls_1600ms+=calls


# Save calls per conditino        
num_calls.to_csv(os.path.join(out_folder, 'callnum_long_short_noise.csv'), index=False)  
# Save calls per condition per eoch
num_calls_p_epoch.to_csv(os.path.join(out_folder, 'callnum_p_epoch_long_short_noise.csv'), index=False)  
            
# Save data for histograms        
hist_data.to_csv(os.path.join(out_folder, 'hist_long_short_noise.csv'), index=False)  


# Save binned data per animal
data = convert_to_R(bins_all, ['session', 'condition', 'time'])
edges_temp = (edges+np.nanmean(np.diff(edges))/2)[:-1]
data['time_sec'] = edges_temp[data['time']]
data['condition'] = np.asarray(conditions)[data['condition']]
data.to_csv(os.path.join(out_folder, 'data_long_short_noise.csv'), index=False)  

# save binned data in perc
data = convert_to_R(bins_all_perc, ['session', 'condition', 'time'])
edges_temp = (edges+np.nanmean(np.diff(edges))/2)[:-1]
data['time_sec'] = edges_temp[data['time']]
data['condition'] = np.asarray(conditions)[data['condition']]
data.to_csv(os.path.join(out_folder, 'data_long_short_noise_perc.csv'), index=False)  
np.save(os.path.join(out_folder,'data_long_short_noise_perc.npy'), bins_all_perc)

data = convert_to_R(np.argmax(bins_all,2), ['session', 'condition'])
data = data[~np.isnan(data).any(axis=1)]
edges_temp = (edges+np.nanmean(np.diff(edges))/2)[:-1]
data['data_sec'] = edges_temp[data['data'].astype(int)]
data['condition'] = np.asarray(conditions)[data['condition']]
data.to_csv(os.path.join(out_folder,'long_short_noise_peak_delay.csv'), index=False)  
