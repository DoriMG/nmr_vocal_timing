# -*- coding: utf-8 -*-
"""
@author: Dori M. Grijseels
"""


import re
import pandas as pd
import numpy as np
import os
import sys
import librosa
from datetime import datetime

# Custom functions import
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'util'))
from preprocessing import find_all_text_files, load_calls, convert_to_R



def call_per_period(start, end, burst_length, df):

    calls_during_noise = []
    current_start = start
    epoch_with_call = []
    while current_start < end:
        current_end = current_start + burst_length*2
        df_temp = df[(df['end']<current_end) & (df['start']>current_start)]
        if len(df_temp)<1:
            calls_during_noise.append([np.nan])
        else:
            calls_during_noise.append(list(df_temp['start'] - current_start))
        current_start = current_end
        
        if len(df_temp) > 0:
            epoch_with_call += [1]
        else:
            epoch_with_call += [0]

    return calls_during_noise, epoch_with_call



def call_timing_per_epoch(start, end, burst_length, df):
    current_start = start

    # For each epoch, find where the call is if there is any
    call_timing = []
    while current_start < end:
        current_end = current_start + burst_length*2
        df_temp = df[(df['end']<current_end) & (df['start']>current_start)]
        call_timing_dich = [0,0]
        for call in list(df_temp['start'] - current_start):
            if call<burst_length:
                call_timing_dich[0]=1
            else:
                call_timing_dich[1]=1
        current_start = current_end
        call_timing.append(call_timing_dich)
            
    return call_timing

def flatten(xss):
    return [x for xs in xss for x in xs]

def get_animal_data(df):
    DOB  = datetime.strptime(df['DOB'].values[0], '%Y-%m-%d')
    rec_date = datetime.strptime(df['rec_date'].values[0], '%Y-%m-%d')
    age = (rec_date-DOB).days
    weight = df['weight'].values[0]
    colony = df['colony'].values[0]
    sex = df['sex'].values[0]
    
    return age, weight, colony, sex
    
    

## Metadata
ani_data_file = os.path.join(os.path.dirname(os.getcwd()), 'metadata', 'animal_data.csv')
ani_data = pd.read_csv(ani_data_file, delimiter=',')


data_folder = os.path.join(os.path.dirname(os.getcwd()), 'data', 'periodic')
out_folder = os.path.join(os.getcwd(), 'data')

all_files = find_all_text_files([data_folder])


# Settings
condition_segments = {}
conditions = ['600ms', '900ms']
period_time = [ 0.6, 0.9]
session_lengths = {} 
edges = np.linspace(0,1.8, 37)
min_tot_calls = 5

# Arrays to save out
bins_all =  np.full([3, len(all_files),len(conditions), 36], np.nan)
hist_data = pd.DataFrame(columns=['data','condition', 'animal'])
num_calls = pd.DataFrame(columns=['data', 'condition', 'animal'])
num_calls_p_epoch = pd.DataFrame(columns=['data', 'condition', 'animal'])

when_early =  np.full([len(all_files),len(conditions), 10], np.nan)


all_calls_300ms = []
all_calls_600ms = []
all_calls_900ms = []


num_calls_p_type= pd.DataFrame(columns=['epoch_type', 'percentage','condition', 'animal'])
stab_across_conditions =  pd.DataFrame(columns=['noise_perc_600', 'noise_perc_900', 'animal'])
stab_within_conditions =  pd.DataFrame(columns=['noise_first_half', 'noise_second_half', 'condition','animal'])
excitability =  pd.DataFrame(columns=['mean_response_noise', 'mean_response_silent', 'percentage','condition','animal'])

# response_delay_early_vs_late
delay_stats = pd.DataFrame(columns=['age', 'condition', 'animal'])
hybrid_count = pd.DataFrame(columns=['age', 'condition', 'animal'])
noise_where = []
for i, file in enumerate(all_files):
    df = load_calls(file)
    filename = os.path.basename(file)
    animal_id = filename.split('_')[0]
    
    df_temp = ani_data[ani_data['Animal'] == int(animal_id)]
    animal_id = str(df_temp['ABN'].values[0])
    
    df_temp = df_temp[df_temp['rec_type']=='noise']
    age, weight, colony, sex = get_animal_data(df_temp)
    
    
    perc_noise = np.full([2,1], np.nan)
    for j,c in enumerate(conditions):
        start_time = df.loc[df['label'] == 'start'+c, 'start'].values[0]
        end_time = df.loc[df['label'] == 'end'+c, 'start'].values[0]
        tot_calls = sum((df['start']>start_time) & (df['end']<end_time))
        
        temp = call_timing_per_epoch(start_time, end_time, period_time[j], df)
        # 4 types of epochs: no call, call during noise, call during silence, call during both
        call_types = np.sum(np.asarray(temp)*[1,2], axis=1)
        call_type_call_epochs = call_types[call_types>0]
        
        calls,epoch_with_call = call_per_period(start_time, end_time, period_time[j], df)

        if tot_calls>min_tot_calls:
            mean_response_time = np.full([3,1], np.nan)
            for epoch_type in np.unique(call_type_call_epochs):
                df_temp = pd.DataFrame.from_dict({'epoch_type': epoch_type, 
                                                  'percentage': [np.nanmean(call_type_call_epochs==epoch_type)],  
                                                  'percentage_all': [np.nanmean(call_types==epoch_type)],
                                                  'condition': [c[:-2]], 
                                                  'animal':[animal_id],
                                                  'experiment':['noise']})
                num_calls_p_type = pd.concat([num_calls_p_type, df_temp])
                calls_temp = []
                for idx in np.where(call_types==epoch_type)[0]:
                    calls_temp += calls[idx]
                bins_all[epoch_type-1, i,j,:] = np.histogram(calls_temp, edges)[0]
                mean_response_time[epoch_type-1] = np.nanmean(calls_temp)
                
                if epoch_type==1:
                    
                    mean_response_time_first = np.nanmean(calls_temp)
                    mean_response_time_second = np.nan
                    mean_response_base = mean_response_time_first
                    mean_response_std = np.nanstd(calls_temp)
                elif epoch_type==2:
                    mean_response_time_first = np.nan
                    mean_response_time_second = np.nanmean(calls_temp)-period_time[j]
                    mean_response_base = mean_response_time_second
                    mean_response_std = np.nanstd([call -period_time[j] for call in calls_temp])
                elif epoch_type == 3:
                    mean_response_time_first = np.nanmean([call for call in calls_temp if call<=period_time[j]])
                    mean_response_time_second = np.nanmean([call-period_time[j] for call in calls_temp if call>period_time[j]])
                    mean_response_base = np.nan
                    mean_response_std = np.nan
                    
                df_temp = pd.DataFrame.from_dict({'mean_response_base': [mean_response_base],
                                                  'mean_response_timing_first': [mean_response_time_first],
                                                  'mean_response_timing_second': [mean_response_time_second],
                                                  'mean_response_std': [mean_response_std],
                                                  'epoch_type': epoch_type, 
                                                  'condition': [c[:-2]], 
                                                  'animal':[animal_id],
                                                  'age': [age], 
                                                  'weight': [weight], 
                                                  'colony':[colony],
                                                  'sex':[sex],
                                                  'experiment':['noise']})
                delay_stats = pd.concat([delay_stats, df_temp])
            perc_noise[j] = np.nanmean(call_type_call_epochs==1)
            
            ## Does the percentage of noise responses correlate with the mean time to call? - i.e. some measure of excitability
            df_temp = pd.DataFrame.from_dict({'mean_response_noise': mean_response_time[0], 
                                              'mean_response_silent': mean_response_time[1]-period_time[j], 
                                              'mean_response_timing': [np.nanmean(flatten(calls))],
                                              'percentage': [np.nanmean(call_type_call_epochs==1)],  
                                              'total': [tot_calls], 
                                              'condition': [c[:-2]], 
                                              'animal':[animal_id],
                                              'age': [age], 
                                              'weight': [weight], 
                                              'colony':[colony],
                                              'sex':[sex],
                                              'experiment':['noise']})
            excitability = pd.concat([excitability, df_temp])
        
            n_onset =  np.nanmean(call_types==1)+np.nanmean(call_types==3)
            n_offset = np.nanmean(call_types==2)+np.nanmean(call_types==3)
            df_temp = pd.DataFrame.from_dict({ 'calls':  [np.nanmean(call_types==3)*100], 
                                             'predicted':[0],
                                              'condition': [c[:-2]], 
                                              'animal':[animal_id],
                                              'age': [age], 
                                              'weight': [weight], 
                                              'colony':[colony],
                                              'sex':[sex],
                                              'experiment':['noise']})
            hybrid_count = pd.concat([hybrid_count, df_temp])
            
            
            df_temp = pd.DataFrame.from_dict( {'calls':  [n_onset*n_offset*100],
                                                 'predicted':[1],
                                              'condition': [c[:-2]], 
                                              'animal':[animal_id],
                                              'age': [age], 
                                              'weight': [weight], 
                                              'colony':[colony],
                                              'sex':[sex],
                                              'experiment':['noise']})
            hybrid_count = pd.concat([hybrid_count, df_temp])
        
            #histogram of when each type takes place
            edges_where =  np.linspace(0,len(call_types), 11)
            when_early[i, j, :] = np.histogram(np.where(call_types==1)[0], edges_where)[0]/np.nansum(call_types==1)
        
        ## Stability within each session - i.e. do the same animals do the same first vs second half? 
        noise_calls_per_half = np.full([2,1], np.nan)
        for half in range(2):
            start_time = df.loc[df['label'] == 'start'+c, 'start'].values[0]
            end_time = df.loc[df['label'] == 'end'+c, 'start'].values[0]
            if half == 0:
                end_time = start_time + (end_time-start_time)/2
            else:
                start_time = start_time + (end_time-start_time)/2
                
            tot_calls = sum((df['start']>start_time) & (df['end']<end_time))
    
            temp = call_timing_per_epoch(start_time, end_time, period_time[j], df)
            # 4 types of epochs: no call, call during noise, call during silence, call during both
            call_types = np.sum(np.asarray(temp)*[1,2], axis=1)
            call_type_call_epochs = call_types[call_types>0]
                
            if tot_calls>min_tot_calls:
                noise_calls_per_half[half] = np.nanmean(call_type_call_epochs==1)
        df_temp = pd.DataFrame.from_dict({'noise_first_half': noise_calls_per_half[0], 
                                           'noise_second_half': noise_calls_per_half[1], 
                                           'condition': [c[:-2]], 
                                           'animal':[animal_id],
                                           'experiment':['noise']})
        stab_within_conditions = pd.concat([stab_within_conditions, df_temp])
                
                
    ## Stability across conditions
    df_temp = pd.DataFrame.from_dict({'noise_perc_600': perc_noise[0], 
                                      'noise_perc_900': perc_noise[1], 
                                      'animal':[animal_id],
                                      'experiment':['noise']})
    stab_across_conditions = pd.concat([stab_across_conditions, df_temp])
    
   
        
    ## Then look at the same in sc/aperiodic - if I have repeat animals, see if I get the same percentages 
        
        
np.save(os.path.join(out_folder, 'bins_by_epoch_type'), bins_all)

# Save binned data per animal
data_noise = convert_to_R(when_early, ['session', 'condition', 'time'])
data_noise['experiment'] = 'noise'

##### Soft chirp experimetns ###########

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

def call_per_stim(start_time, stim_times, burst_length, df):
    epoch_with_call = []
    calls_during_noise = []
    for stim in stim_times:
        current_start = stim + start_time
        current_end = current_start + (burst_length*2)
        df_temp = df[(df['end']<current_end) & (df['start']>current_start)]
        if len(df_temp)<1:
            calls_during_noise.append([np.nan])
        else:
            calls_during_noise.append(list(df_temp['start'] - current_start))
             
        if len(df_temp) > 0:
            epoch_with_call += [1]
        else:
            epoch_with_call += [0]

    return calls_during_noise, epoch_with_call



def call_timing_per_stim(start_time, stim_times, burst_length, df):
    call_timing = []
    for stim in stim_times:
        current_start = stim + start_time
        current_end = current_start + (burst_length*2)
        df_temp = df[(df['end']<current_end) & (df['start']>current_start)]
        call_timing_dich = [0,0]
        for call in list(df_temp['start'] - current_start):
            if call<burst_length:
                call_timing_dich[0]=1
            else:
                call_timing_dich[1]=1
        call_timing.append(call_timing_dich)
            
    return call_timing


data_folder = os.path.join(os.path.dirname(os.getcwd()), 'data', 'sc')
stim_folder = os.path.join(os.path.dirname(os.getcwd()), 'metadata', 'stimuli')
out_folder = os.path.join(os.getcwd(), 'data')

all_files = find_all_text_files([data_folder])


# Settings
condition_segments = {}
conditions = ['600sc', '900sc']
period_time = [0.6, 0.9]
session_lengths = {} 
edges = np.linspace(0,1.8, 37)
min_tot_calls = 5

# Arrays to save out
bins_all =  np.full([len(all_files),len(conditions), 36], np.nan)
hist_data = pd.DataFrame(columns=['data','condition', 'animal'])
num_calls = pd.DataFrame(columns=['data', 'condition', 'animal'])
num_calls_p_epoch = pd.DataFrame(columns=['data', 'condition', 'animal'])

all_calls_300ms = []
all_calls_600ms = []
all_calls_900ms = []

# Control for inaccuracies in stimulus creation
stim_time = get_start_times(conditions, stim_folder)
when_early =  np.full([len(all_files),len(conditions), 10], np.nan)


## Run through folders

for i, file in enumerate(all_files):
    df = load_calls(file)
    filename = os.path.basename(file)
    animal_id = filename.split('_')[0][3:-4]
    
    df_temp = ani_data[ani_data['ABN'] == int(animal_id)]
    
    df_temp = df_temp[df_temp['rec_type']=='sc']
    age, weight, colony, sex = get_animal_data(df_temp)
    
    perc_noise = np.full([2,1], np.nan)
    for j,c in enumerate(conditions):
        start_time = df.loc[df['label'] == 'start_'+c, 'start'].values[0]
        end_time = df.loc[df['label'] == 'end_'+c, 'start'].values[0]
        
        # get the total number of calls in this period, check if more than a ceratin minimum
        tot_calls = sum((df['start']>start_time) & (df['end']<end_time))
        df_temp = pd.DataFrame.from_dict({'data': [tot_calls], 'condition': [c], 'animal': [animal_id]})
        num_calls = pd.concat([num_calls, df_temp])
        calls,epoch_with_call = call_per_stim(start_time, stim_time[c], period_time[j], df)
            
        temp = call_timing_per_stim(start_time, stim_time[c], period_time[j], df)
        # 4 types of epochs: no call, call during noise, call during silence, call during both
        call_types = np.sum(np.asarray(temp)*[1,2], axis=1)
        call_type_call_epochs = call_types[call_types>0]
        
        if tot_calls>min_tot_calls:
            mean_response_time = np.full([3,1], np.nan)
            for epoch_type in np.unique(call_type_call_epochs):
                df_temp = pd.DataFrame.from_dict({'epoch_type': epoch_type, 
                                                  'percentage': [np.nanmean(call_type_call_epochs==epoch_type)],  
                                                  'percentage_all': [np.nanmean(call_types==epoch_type)],
                                                  'condition': [c[:-2]], 
                                                  'animal':[animal_id],
                                                  'experiment':['sc']})
                
                
                num_calls_p_type = pd.concat([num_calls_p_type, df_temp])
                calls_temp = []
                for idx in np.where(call_types==epoch_type)[0]:
                    calls_temp += calls[idx]
                # bins_all[epoch_type-1, i,j,:] = np.histogram(calls_temp, edges)[0]
                mean_response_time[epoch_type-1] = np.nanmean(calls_temp)
                
                if epoch_type==1:
                    
                    mean_response_time_first = np.nanmean(calls_temp)
                    mean_response_time_second = np.nan
                    mean_response_base = mean_response_time_first
                    mean_response_std = np.nanstd(calls_temp)
                elif epoch_type==2:
                    mean_response_time_first = np.nan
                    mean_response_time_second = np.nanmean(calls_temp)-period_time[j]
                    mean_response_base = mean_response_time_second
                    mean_response_std = np.nanstd(calls_temp)
                elif epoch_type == 3:
                    mean_response_time_first = np.nanmean([call for call in calls_temp if call<=period_time[j]])
                    mean_response_time_second = np.nanmean([call-period_time[j] for call in calls_temp if call>period_time[j]])
                    mean_response_base = np.nan
                    mean_response_std = np.nan
                    
                df_temp = pd.DataFrame.from_dict({'mean_response_base': [mean_response_base],
                                                  'mean_response_timing_first': [mean_response_time_first],
                                                  'mean_response_timing_second': [mean_response_time_second],
                                                  'mean_response_std': [mean_response_std],
                                                  'epoch_type': epoch_type, 
                                                  'condition': [c[:-2]], 
                                                  'animal':[animal_id],
                                                  'age': [age], 
                                                  'weight': [weight], 
                                                  'colony':[colony],
                                                  'sex':[sex],
                                                  'experiment':['sc']})
                delay_stats = pd.concat([delay_stats, df_temp])
            # perc_noise[j] = np.nanmean(call_type_call_epochs==1)
            
            ## Does the percentage of noise responses correlate with the mean time to call? - i.e. some measure of excitability
            df_temp = pd.DataFrame.from_dict({'mean_response_noise': mean_response_time[0], 
                                              'mean_response_silent': mean_response_time[1]-period_time[j], 
                                              'mean_response_timing': [np.nanmean(flatten(calls))],
                                              'percentage': [np.nanmean(call_type_call_epochs==1)],  
                                              'total': [tot_calls],  
                                              'condition': [c[:-2]], 
                                              'animal':[animal_id],
                                              'age': [age], 
                                              'weight': [weight], 
                                              'colony':[colony],
                                              'sex':[sex],
                                              'experiment':['sc']})
            excitability = pd.concat([excitability, df_temp])
            perc_noise[j] = np.nanmean(call_type_call_epochs==1)
            
           
            n_onset =  np.nanmean(call_types==1)+np.nanmean(call_types==3)
            n_offset = np.nanmean(call_types==2)+np.nanmean(call_types==3)
            df_temp = pd.DataFrame.from_dict({ 'calls':  [np.nanmean(call_types==3)*100], 
                                             'predicted':[0],
                                              'condition': [c[:-2]], 
                                              'animal':[animal_id],
                                              'age': [age], 
                                              'weight': [weight], 
                                              'colony':[colony],
                                              'sex':[sex],
                                              'experiment':['sc']})
            hybrid_count = pd.concat([hybrid_count, df_temp])
            
            
            df_temp = pd.DataFrame.from_dict( {'calls':  [n_onset*n_offset*100],
                                                 'predicted':[1],
                                              'condition': [c[:-2]], 
                                              'animal':[animal_id],
                                              'age': [age], 
                                              'weight': [weight], 
                                              'colony':[colony],
                                              'sex':[sex],
                                              'experiment':['sc']})
            hybrid_count = pd.concat([hybrid_count, df_temp])
            
            #histogram of when each type takes place
            edges_where =  np.linspace(0,len(call_types), 11)
            when_early[i, j, :] =  np.histogram(np.where(call_types==1)[0], edges_where)[0]/np.nansum(call_types==1)
        
        ## stability
        ## Stability within each session - i.e. do the same animals do the same first vs second half? 
        noise_calls_per_half = np.full([2,1], np.nan)
        for half in range(2):
            start_time = df.loc[df['label'] == 'start_'+c, 'start'].values[0]
            end_time = df.loc[df['label'] == 'end_'+c, 'start'].values[0]
            temp = stim_time[c]
            if half == 0:
                stim_times = temp[:int(np.floor(len(temp)/2))]
            else:
                start_time = temp[int(np.floor(len(temp)/2))]
                temp =  temp[int(np.floor(len(temp)/2)):]
                stim_times = [x-temp[0] for x in temp]
                     
            temp = call_timing_per_stim(start_time, stim_times, period_time[j], df)
            # 4 types of epochs: no call, call during noise, call during silence, call during both
            call_types = np.sum(np.asarray(temp)*[1,2], axis=1)
            call_type_call_epochs = call_types[call_types>0]
                
            if tot_calls>min_tot_calls:
                noise_calls_per_half[half] = np.nanmean(call_type_call_epochs==1)
        df_temp = pd.DataFrame.from_dict({'noise_first_half': noise_calls_per_half[0], 
                                           'noise_second_half': noise_calls_per_half[1], 
                                          'condition': [c[:-2]], 
                                           'animal':[animal_id],
                                           'experiment':['sc']})
        stab_within_conditions = pd.concat([stab_within_conditions, df_temp])
        
    ## Stability across conditions
    df_temp = pd.DataFrame.from_dict({'noise_perc_600': perc_noise[0], 
                                      'noise_perc_900': perc_noise[1], 
                                      'animal':[animal_id],
                                      'experiment':['sc']})
    stab_across_conditions = pd.concat([stab_across_conditions, df_temp])


# Save calls per condition 
num_calls_p_type.to_csv(os.path.join(out_folder, 'strategy_per_animal.csv'), index=False)  

stab_across_conditions.to_csv(os.path.join(out_folder, 'stab_across_conditions.csv'), index=False)  

stab_within_conditions.to_csv(os.path.join(out_folder, 'stab_within_conditions.csv'), index=False) 

excitability.to_csv(os.path.join(out_folder, 'excitability.csv'), index=False) 

delay_stats.to_csv(os.path.join(out_folder, 'delay_stats.csv'), index=False) 

hybrid_count.to_csv(os.path.join(out_folder, 'hybrid_count.csv'), index=False) 
#
data_sc = convert_to_R(when_early, ['session', 'condition', 'time'])
data_sc['experiment'] = 'sc'

data = pd.concat([data_noise, data_sc])
data.to_csv(os.path.join(out_folder, 'timing_in_session.csv'), index=False) 
