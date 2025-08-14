# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 20:25:38 2025

@author: Dori M. Grijseels
"""


import numpy as np
import os
import pandas as pd

# Custom functions import
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'util'))
from preprocessing import load_calls, convert_to_R

def calculate_interruptions(df):
    interrupt = []
    for index, row in df.iterrows():
        if index>0:
            if row.label == 's':
                if row.start<df.iloc[index-1].end:
                    interrupt.append(1)
                else:
                    interrupt.append(0)
    return np.nanmean(interrupt)

def calc_ici(df):
    ici = []
    for index, row in df.iterrows():
        if index>0:
            # Check current call is focal animal, and last call is non-vocal
            if row.label == 's':
                if df.iloc[index-1].label == 'c':
                    ici.append(row.start-df.iloc[index-1].end)
    return ici

def get_ici_across_calls(all_files, n_bins, bins):
    bs = ['Passing', 'Nose-to-nose', 'Body-to-body contact', 'Nose-to-body contact',
       'Anogenital sniffing', 'Other interaction']
    all_ici = []
    all_ici_by_ani = np.full([6,len(all_files)], np.nan)
    probs = np.zeros([all_files.__len__(), 6, n_bins-1])

    for f, file in enumerate(all_files):
        calls = load_calls(all_files[f])
        calls = calls[calls['label'].isnull()]
        calls = calls[['start','end']].to_numpy()
        dataset = os.path.splitext(all_files[f])[0]
        _, fname = os.path.split(dataset)
        ici_by_ani = []
        
        if calls.size > 0:
        # find the behavior file
            if os.path.isfile(os.path.join(behavior_folder, fname+'.csv')):
                print(fname)
                behav_file = os.path.join(behavior_folder, fname+'.csv')
                df = pd.read_csv(behav_file)
                df[['behavior_code']] = df[['Behavior']].apply(lambda col:pd.Categorical(col).codes)

                behavior_array = df[["behavior_code","Start (s)","Stop (s)"]].values
                behavior_strings = df[['Behavior']].values

                # Average calls from onset of behavior
                
                for b in range(6):
                    all_ici_temp = []
                    behavior_count = 0
                    for i, behavior in enumerate(behavior_array):
                        if behavior_strings[i] == bs[b]: # nose to nose
                            ici_temp = calls[:,0] - behavior[1]
                            all_ici_temp = np.append(all_ici_temp, ici_temp)
                            behavior_count += 1
                            first_ici =  [x for x in all_ici_temp if x>0]
                            first_ici =  [x for x in first_ici if x<3]
                            if len(first_ici)>0:
                                ici_by_ani += [first_ici[0]]

                    all_ici_inc = [x for x in all_ici_temp if x>-10 ]
                    all_ici_inc = [x for x in all_ici_inc if x<10 ]
                    all_ici = np.append(all_ici, all_ici_inc)
                    
                    all_ici_by_ani[b, f] = np.nanmean(ici_by_ani)

                    
                    vals, edges = np.histogram(all_ici_temp, bins=bins)
                    if behavior_count > 0:
                        probs[f,b,:] = vals/behavior_count
                    
    return probs, all_ici_by_ani

# Data folders two_animals
data_folder = os.path.join(os.path.dirname(os.getcwd()), 'data')
call_folder = os.path.join(data_folder, 'hierarchy', 'labels')
id_folder = os.path.join(data_folder, 'hierarchy', 'id_labels')
behavior_folder = os.path.join(data_folder, 'hierarchy', 'boris')
call_folder_single =  os.path.join(data_folder, 'playback')
save_folder = os.path.join(os.getcwd(), 'data')

# Get list of files
all_files = []
for file in os.listdir(id_folder):
    if file.endswith(".txt"):
        all_files.append(os.path.join(id_folder, file))

# Figure A-F calculate ICI and interruptions

# Split into two animals
computer_calls = []
animal_calls = []  
for f, file in enumerate(all_files):
    calls = load_calls(all_files[f])
    calls = calls[calls['label']!='?']
    calls = calls[calls['label']!='n'] # n used to mark noise instances
    calls = calls[calls['label'].str.len()==1]
    calls = calls[calls['label'].notnull()]
    unique_ids = calls['label'].unique()
    call_id = calls['label'].to_numpy()
    calls = calls[['start','end']].to_numpy()
    for i, ani_id in enumerate(unique_ids):
        temp_calls = [c for i,c in enumerate(calls) if call_id[i]==ani_id]
        df = pd.DataFrame(temp_calls, columns=['start', 'end'])
        if i == 0:
            df['label'] = 's'
            animal_calls.append(df)
        else:
            df['label'] = 'c'
            computer_calls.append(df)
            
            
# Loop through combinations and add to shuffle and real data
interruptions_shuff = []
interruptions = []
all_ici = pd.DataFrame(columns=['ici', 'file_num', 'df_playback'])
 # will save ici for both experiments
for i, anim_df in enumerate(animal_calls):
    for j, comp_df in enumerate(computer_calls):
        result = pd.concat([anim_df, comp_df])
        result = result.sort_values(by=['start'],ignore_index=True)
        interruption_rate = calculate_interruptions(result)
        if i==j:
            interruptions.append(interruption_rate)
            # Calculate ici
            ici = calc_ici(result)
            df = pd.DataFrame(ici, columns=['ici'])
            df['file_num'] = i
            df['playback'] = 0
            all_ici = pd.concat([all_ici, df], ignore_index=True)
            
            # Check animal 2 interrupting animal 1
            anim_df['label'] ='c'
            comp_df['label'] = 's'
            result = pd.concat([anim_df, comp_df])
            result = result.sort_values(by=['start'],ignore_index=True)
            interruption_rate = calculate_interruptions(result)
            interruptions.append(interruption_rate)
            
            # Calculate ici
            ici = calc_ici(result)
            df = pd.DataFrame(ici, columns=['ici'])
            df['file_num'] = i
            df['playback'] = 0
            all_ici = pd.concat([all_ici, df], ignore_index=True)
        else:
            interruptions_shuff.append(interruption_rate)
       
      

df_hier = pd.DataFrame({'data':interruptions, 'shuff': 0, 'playback': 0})
df_hier_shuff = pd.DataFrame({'data':interruptions_shuff, 'shuff': 1, 'playback': 0})


## Interruptions single animal

# Get list of files
all_files = []
for fold in os.listdir(call_folder_single):
    for file in os.listdir(os.path.join(call_folder_single, fold)):
        if file.endswith(".txt"):
            all_files.append(os.path.join(call_folder_single,fold, file))
        
computer_calls = []
animal_calls = []
for file in all_files:
    calls = load_calls(file)
    computer_calls.append(calls[calls['label']=='c'])
    ani_calls = calls[calls['label'].isnull()]
    ani_calls['label'] = 's'
    animal_calls.append(ani_calls)
      
interruptions_shuff = []
interruptions = []
interrupt_time = []
interrupt_time_shuff = []
for i, anim_df in enumerate(animal_calls):
    for j, comp_df in enumerate(computer_calls):
        result = pd.concat([anim_df, comp_df])
        result = result.sort_values(by=['start'],ignore_index=True)
        interruption_rate = calculate_interruptions(result)
        if i==j:
            interruptions.append(interruption_rate)
            
            # Calculate ici
            ici = calc_ici(result)
            df = pd.DataFrame(ici, columns=['ici'])
            df['file_num'] = i
            df['playback'] = 1
            all_ici = pd.concat([all_ici, df], ignore_index=True)
        else:
            interruptions_shuff.append(interruption_rate)

# Save out ICI
all_ici.to_csv(os.path.join(save_folder, 'all_ici.csv'), index=False) 
              

# Save out as one dataset
df_pb = pd.DataFrame({'data':interruptions, 'shuff': 0, 'playback': 1})
df_pb_shuff = pd.DataFrame({'data':interruptions_shuff, 'shuff': 1, 'playback': 1})
df_all = pd.concat([df_pb, df_pb_shuff, df_hier, df_hier_shuff], ignore_index=True, sort=False)
df_all.to_csv(os.path.join(save_folder, 'interruptions.csv'), index=False) 
       

    
# Figure G-J       
# Set parameters for binning
n_bins = 46
bins = np.linspace(0, 15, n_bins)-5

# Get responses relative to behavior
probs, all_ici_by_ani = get_ici_across_calls(all_files, n_bins, bins)

# Save out responses relative to behaviro
bs = ['Passing', 
       'Snout-to-snout contact', 
       'Body-to-body contact', 
       'Snout-to-body contact',
        'Anogenital sniffing', 
        'Other interaction']


data = convert_to_R(probs, ['session', 'touch', 'time'])
edges_temp = (bins+np.nanmean(np.diff(bins))/2)[:-1]
data['time_sec'] = edges_temp[data['time']]
tts_temp = np.asarray(bs)
data['touch_type'] = tts_temp[data['touch']]

data.to_csv(os.path.join(save_folder,'calls_after_behavior.csv'), index=False)  




all_call_during_behavior = pd.DataFrame(columns=['during_behavior', 'call_during', 'time_from_behavior', 'call_time', 'session'])
perc_calls_during_behavior = pd.DataFrame(columns=['perc', 'session', 'shuffle', 'call_file', 'behavior_file'])
perc_calls_by_behavior = pd.DataFrame(columns=['Body-to-body contact',
                                               'Nose-to-body contact',
                                               'Nose-to-nose',
                                               'Passing',
                                               'Anogenital sniffing',
                                               'Other interaction'])
for f, file in enumerate(all_files):
    calls = load_calls(all_files[f])
    calls = calls[calls['label'].isnull()]
    calls = calls[['start','end']].to_numpy()
    dataset = os.path.splitext(all_files[f])[0]
    _, fname = os.path.split(dataset)
    
    call_during_behavior = pd.DataFrame(columns=['during_behavior', 'call_during', 'time_from_behavior', 'call_time'])
    if calls.size > 0:
    # find the behavior file
        if os.path.isfile(os.path.join(behavior_folder, fname+'.csv')):
            print(fname)
            testfile = os.path.join(behavior_folder, fname+'.csv')
            df = pd.read_csv(testfile)
            df[['behavior_code']] = df[['Behavior']].apply(lambda col:pd.Categorical(col).codes)
    
            behavior_array = df[["behavior_code","Start (s)","Stop (s)"]].values
            behavior_strings = df[['Behavior']].values
            behavior_array = behavior_array[(behavior_strings!='Genital licking').flatten(),:]
            # Average calls from onset of behavior
            
            for call in calls:
                # call starts during a behavior
                inc = np.logical_and(call[0]>behavior_array[:,1], call[0]<behavior_array[:,2])
                
                call_type_during = np.nan
                if np.any(inc):
                    call_type_during = behavior_strings[np.where(inc)[0][0]]
                
                # call start relative to closest behavior start
                call_start_off = np.min(np.abs(call[0]-behavior_array[:,1]))
                call_type_start = behavior_strings[np.argmin(np.abs(call[0]-behavior_array[:,1]))]
                
                df_temp = pd.DataFrame.from_dict({'during_behavior': int(np.any(inc)), 
                                                  'call_during': call_type_during, 
                                                  'time_from_behavior': call_start_off, 
                                                  'call_time': call_type_start})
            
                call_during_behavior = pd.concat([call_during_behavior, df_temp])
            
            all_call_during_behavior = pd.concat([all_call_during_behavior, call_during_behavior])

            
            count_by_behave_type = call_during_behavior['call_during'].value_counts(dropna=False, normalize=True)
            perc_calls_by_behavior = pd.concat([perc_calls_by_behavior, pd.DataFrame(count_by_behave_type).transpose()])
            
            
            df_temp = pd.DataFrame.from_dict({'perc': np.mean(call_during_behavior['during_behavior']), 
                                              'session': [f],
                                              'shuffle': 'Data',
                                              'call_file': fname,
                                              'behavior_file': fname})
            
            perc_calls_during_behavior = pd.concat([perc_calls_during_behavior, df_temp])



### run shuffles for Fig G
for f, file in enumerate(all_files):
   
    for shuff in range(len(all_files)):
        if shuff != f:
            calls = load_calls(all_files[f])
            calls = calls[calls['label'].isnull()]
            calls = calls[['start','end']].to_numpy()
            
            dataset = os.path.splitext(all_files[f])[0]
            _, fname_ori = os.path.split(dataset)
            
            
            dataset = os.path.splitext(all_files[shuff])[0]
            _, fname = os.path.split(dataset)
            
            
            call_during_behavior = pd.DataFrame(columns=['during_behavior', 'call_during', 'time_from_behavior', 'call_time'])
            if calls.size > 0:
            # find the behavior file
                if os.path.isfile(os.path.join(behavior_folder, fname+'.csv')):
                    print(fname)
                    testfile = os.path.join(behavior_folder, fname+'.csv')
                    df = pd.read_csv(testfile)
                    df[['behavior_code']] = df[['Behavior']].apply(lambda col:pd.Categorical(col).codes)
            
                    behavior_array = df[["behavior_code","Start (s)","Stop (s)"]].values
                    behavior_strings = df[['Behavior']].values
                    behavior_array =  behavior_array[(behavior_strings!='Genital licking').flatten(),:]
            
                    # Average calls from onset of behavior
                    
                    for call in calls:
                        # call starts during a behavior
                        inc = np.logical_and(call[0]>behavior_array[:,1], call[0]<behavior_array[:,2])
                        
                        call_type_during = np.nan
                        if np.any(inc):
                            call_type_during = behavior_strings[np.where(inc)[0][0]]
                        
                        # call start relative to closest behavior start
                        call_start_off = np.min(np.abs(call[0]-behavior_array[:,1]))
                        call_type_start = behavior_strings[np.argmin(np.abs(call[0]-behavior_array[:,1]))]
                        
                        df_temp = pd.DataFrame.from_dict({'during_behavior': int(np.any(inc)), 
                                                          'call_during': call_type_during, 
                                                          'time_from_behavior': call_start_off, 
                                                          'call_time': call_type_start})
                    
                        call_during_behavior = pd.concat([call_during_behavior, df_temp])
                    
                    
                    df_temp = pd.DataFrame.from_dict({'perc': np.mean(call_during_behavior['during_behavior']), 
                                                      'session': [f],
                                                      'shuffle': 'Shuffle',
                                                      'call_file': fname_ori,
                                                      'behavior_file': fname})
                    
                    perc_calls_during_behavior = pd.concat([perc_calls_during_behavior, df_temp])

all_call_during_behavior.to_csv(os.path.join(save_folder, 'all_call_during_behavior.csv'), index=False)  
perc_calls_during_behavior.to_csv(os.path.join(save_folder,'perc_calls_during_behavior.csv'), index=False)  


perc_calls_by_behavior_array = perc_calls_by_behavior.values
behavior_key = perc_calls_by_behavior.keys()

data = convert_to_R(perc_calls_by_behavior_array, ['session', 'touch'])
tts_temp = np.asarray(behavior_key)
data['touch_type'] = tts_temp[data['touch']]
data['data'] = data['data'].fillna(0)
data['touch_type'] = data['touch_type'].fillna('No touch')
data = data.drop(data[data.touch_type =='Genital licking'].index)
data.to_csv(os.path.join(save_folder,'perc_calls_by_behavior.csv'), index=False)  
