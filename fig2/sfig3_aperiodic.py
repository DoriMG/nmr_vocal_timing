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

def process_timestamps(file_path):
    #PROCESSES .TXT FILE WITH SOFT CHIRPS (CAN BE ALONE OR WITH OTHER EXPERIMENTAL RESULTS)
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['start', 'end', 'label'])
    df['label'] = df['label'].str.strip()
    
    first_start_value = df['start'].iloc[0]
    df['start'] = df['start'] - first_start_value
    df['end'] = df['end'] - first_start_value

    return df


def extract_key(file_path):
    #HELPER FUNCTION FOR PROCESSING FILES
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    key = parts[0]  
    return key


def classify_duration(duration_ms):
    #HELPER FUNCTION TO CLASSIFY THE DURATION TYPE IN THE ANALYZE_AUDIO FUNCTION
    if 290 <= duration_ms <= 310:
        return '300'
    elif 590 <= duration_ms <= 610:
        return '600'
    elif 890 <= duration_ms <= 910:
        return '900'
    else:
        return None

def get_stimuli_df(path):
    y, sr = librosa.load(path, sr=None)

    print(librosa.get_duration(y=y, sr=sr))
    frame_length = int(sr * 0.001)
    hop_length = frame_length

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    silence_threshold = 0.01

    is_silent = rms < silence_threshold

    silent_diff = np.diff(is_silent.astype(int))

    start_silence = np.where(silent_diff==1)[0]
    start_noise = np.where(silent_diff==-1)[0]
    start_noise = np.insert(start_noise, 0, 0)
    start_noise = np.append(start_noise, len(is_silent))

    for i, noise_start in enumerate(start_noise[:-1]):
        start_time = noise_start
        end_time = start_noise[i+1]
        silent_start = start_silence[i]

        noise_duration = classify_duration(silent_start-start_time)
        silence_duration = classify_duration(end_time-silent_start)

    fs = sr/frame_length

    df_all = pd.DataFrame(columns=['start_time','end_time', 'silent_start', 'noise_duration', 'silence_duration', 'condition'])
    for i, noise_start in enumerate(start_noise[:-1]):
        start_time = noise_start
        end_time = start_noise[i+1]
        silent_start = start_silence[i]

        noise_duration = classify_duration(silent_start-start_time)
        silence_duration = classify_duration(end_time-silent_start)

        df_temp = pd.DataFrame({'start_time': start_time/fs,
                                'end_time': end_time/fs,
                                'silent_start': silent_start/fs,
                                'noise_duration':noise_duration,
                                'silence_duration':silence_duration,
                                'condition': noise_duration+'_'+silence_duration}, index=[0])
        df_all = pd.concat([df_all, df_temp])
    return df_all

def calls_per_condition(df, df_all):
    result = {
            "300_300": [],
            "300_600": [],
            "300_900": [],
            "600_300": [],
            "600_600": [],
            "600_900": [],
            "900_300": [],
            "900_600": [],
            "900_900": [],
        }
    
    tot_pres = {
            "300_300": 0,
            "300_600": 0,
            "300_900": 0,
            "600_300": 0,
            "600_600": 0,
            "600_900": 0,
            "900_300": 0,
            "900_600": 0,
            "900_900": 0,
        }

    for i, row in df_all.iterrows():
        condition = row['condition']
        start_time = row['start_time']
        end_time = row['end_time']
        df_temp = df[(df['end']<end_time) & (df['start']>start_time)]
        result[condition]+=  list(df_temp['start'] - start_time)
        tot_pres[condition]+=1

    return result, tot_pres

def convert_to_R_AP(result, edges, animal):
    df_R = pd.DataFrame(columns=['value','time', 'dataset', 'condition', 'noise', 'silence'])
    conditions = list(result.keys())
    for c in conditions:
        array_vals = result[c]
        call_bins = np.histogram(array_vals, edges)[0]
        noise_silence_times = c.split("_")
        noise_time = noise_silence_times[0]
        silence_time = noise_silence_times[1]
        df_temp = pd.DataFrame({'value': call_bins,
                                    'time': (edges+0.025)[:-1], 
                                    'dataset': animal,
                                    'condition': c, 
                                    'noise': noise_time,
                                    'silence': silence_time})
        df_R = pd.concat([df_R, df_temp])
    return df_R


# Data folders 
data_folder = os.path.join(os.path.dirname(os.getcwd()), 'data', 'aperiodic')
stim_folder = os.path.join(os.path.dirname(os.getcwd()), 'metadata', 'aperiodic_stimuli')
out_folder = os.path.join(os.getcwd(), 'data')


# Load data files
all_files = find_all_text_files([data_folder])

# divide into two conditions
predictable_data = {}
unpredictable_data = {}

for file in all_files:
    ani_id = extract_key(file)
    df = process_timestamps(file)
    if 'AP' in file:
        predictable_data[f'labels_ap_{ani_id}'] = df
    elif 'AUP' in file:
        unpredictable_data[f'labels_aup_{ani_id}'] = df

# Get stimuli timing
unpredictable_stimulus = 'unpredictable_stimulus_20240701.wav'
predictable_stimulus = 'aperiodic_stimulus_20240701.wav'

df_unpred = get_stimuli_df(os.path.join(stim_folder, unpredictable_stimulus))
df_pred = get_stimuli_df(os.path.join(stim_folder, predictable_stimulus))   


# Predictable aperiodic (S3A-F)
edges = np.linspace(0,1.8, 37)
conditions = df_pred['condition'].unique()

hist_data = pd.DataFrame(columns=['data','condition', 'animal'])
num_calls = pd.DataFrame(columns=['data', 'condition', 'animal'])

df_R_all  = pd.DataFrame(columns=['value','time', 'dataset', 'condition', 'noise', 'silence'])
for i, df in enumerate(predictable_data.values()): 
    df = df[df['label'].isnull()]

    result, tot_pres = calls_per_condition(df, df_pred)
    df_R = convert_to_R_AP(result, edges, list(predictable_data.keys())[i])
    df_R_all = pd.concat([df_R_all, df_R])
    
    for j,c in enumerate(conditions):
        tot_calls = len(result[c])/tot_pres[c]
        df_temp = pd.DataFrame.from_dict({'data': [tot_calls], 'condition': [c], 'animal': list(predictable_data.keys())[i]})
        num_calls = pd.concat([num_calls, df_temp])
        
        if len(result[c])>0:
            df_temp = pd.DataFrame.from_dict({'data': result[c]})
            df_temp['condition'] = c
            df_temp['animal'] = list(predictable_data.keys())[i]
            hist_data = pd.concat([hist_data, df_temp])
            
df_R_all.to_csv(os.path.join(out_folder, 'results_AP_predictable.csv'), index=False)  

# Save calls per conditino        
num_calls.to_csv(os.path.join(out_folder, 'callnum_AP_predictable.csv'), index=False)  

# Save data for histograms        
hist_data.to_csv(os.path.join(out_folder, 'hist_AP_predictable.csv'), index=False)  

new_df = pd.DataFrame({'max_idx' : df_R_all.groupby( ['dataset', 'noise'] )['value'].idxmax()}).reset_index()
exc = pd.DataFrame({'sum' : df_R_all.groupby( ['dataset', 'noise'] )['value'].sum()}).reset_index()
new_df['exc'] = exc['sum'] == 0

new_df = new_df[new_df.exc == False]
edges_temp = (edges+np.nanmean(np.diff(edges))/2)[:-1]
new_df['data_sec'] = edges_temp[new_df['max_idx'].astype(int)]
new_df.to_csv(os.path.join(out_folder, 'AP_peak_delay.csv'), index=False)  


# Unpredictable aperiodic (S3G-L)
edges = np.linspace(0,1.8, 37)
conditions = df_unpred['condition'].unique()

hist_data = pd.DataFrame(columns=['data','condition', 'noise', 'silence', 'animal'])
num_calls = pd.DataFrame(columns=['data', 'condition', 'noise', 'silence','animal'])

df_R_all  = pd.DataFrame(columns=['value','time', 'dataset', 'condition', 'noise', 'silence'])
for i, df in enumerate(unpredictable_data.values()): 
    df = df[df['label'].isnull()]

    result, tot_pres = calls_per_condition(df, df_unpred)
    df_R = convert_to_R_AP(result, edges, list(unpredictable_data.keys())[i])
    df_R_all = pd.concat([df_R_all, df_R])
    

    
    for j,c in enumerate(conditions):
        cc = c.split('_')
        if tot_pres[c]>0:
            tot_calls = len(result[c])/tot_pres[c]
        else:
            tot_calls = 0
        df_temp = pd.DataFrame.from_dict({'data': [tot_calls], 'condition': [c], 'noise': cc[0], 'silence': cc[1],'animal': list(unpredictable_data.keys())[i]})
        num_calls = pd.concat([num_calls, df_temp])
        
        if tot_calls>0:
            df_temp = pd.DataFrame.from_dict({'data': result[c]})
            df_temp['condition'] = c
            df_temp['noise'] = cc[0]
            df_temp['silence'] = cc[1]
            df_temp['animal'] = list(unpredictable_data.keys())[i]
            hist_data = pd.concat([hist_data, df_temp])
            
df_R_all.to_csv(os.path.join(out_folder, 'results_AUP.csv'), index=False)  

# Save calls per conditino        
num_calls.to_csv(os.path.join(out_folder, 'callnum_AUP.csv'), index=False)  

# Save data for histograms        
hist_data.to_csv(os.path.join(out_folder, 'hist_AUP.csv'), index=False)  

new_df = pd.DataFrame({'max_idx' : df_R_all.groupby( ['dataset', 'noise'] )['value'].idxmax()}).reset_index()
exc = pd.DataFrame({'sum' : df_R_all.groupby( ['dataset', 'noise'] )['value'].sum()}).reset_index()
new_df['exc'] = exc['sum'] == 0

new_df = new_df[new_df.exc == False]
edges_temp = (edges+np.nanmean(np.diff(edges))/2)[:-1]
new_df['data_sec'] = edges_temp[new_df['max_idx'].astype(int)]
new_df.to_csv(os.path.join(out_folder, 'AUP_peak_delay.csv'), index=False)  
