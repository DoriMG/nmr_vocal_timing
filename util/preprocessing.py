# -*- coding: utf-8 -*-
"""
@author: Dori M. Grijseels
"""

import pandas as pd
import os


def find_all_text_files(folders, check_sub_folders=1):
    """Find all files with a .txt extension in all folders in folder list (and subfolders if check_sub_folders = 1 (default))"""
    all_files = []
    for f, folder in enumerate(folders):
        for subfolder in os.listdir(folder):

            if os.path.isdir(os.path.join(folder, subfolder)):
                if check_sub_folders:
                    folders.append(os.path.join(folder, subfolder))                
            elif subfolder.endswith(".txt"):
                all_files.append(os.path.join(folder,subfolder))
    return all_files

def load_calls(file, delimiter='\t'):
    """Extract calls from txt outputed by Audacity.
    
    If you only want a certain type of call, set filter_calls to 1, and input the name of the call_type as it is named in the txt file (e.g. 's').

    """
    # read calls into pandas dataframe
    calls = pd.read_csv(file, delimiter='\t', names = ['start', 'end', 'label'])
    
    return calls

def convert_to_R(data, columns):
    shape = data.shape
    index = pd.MultiIndex.from_product([range(s)for s in shape], names=columns)
    df = pd.DataFrame({'data': data.flatten()}, index=index).reset_index()
    return df