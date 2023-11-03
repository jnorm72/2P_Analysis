# -*- coding: utf-8 -*-
"""
Aggregates the HDF5 files and extracts the desired firing rates

@author: jnorm
"""

# 
# expects HDF5 files with mCherry labels, will save numpy array
"""
Aggregates data across mice and calculates cell firing rates for different behaviors
    Data should already be cleaned and have licks, stims, and mCherry added

::Inputs::   
  path to hdf5 file
    'time' : vector of times that are still present after cleaning
    'spikes' : array (cells x time) that indicates when a cell fires
    'stims_binary' : vector of times that are during a stimulation block (0 or 1)
    'licks' : vector of times when the mouse licks (0 or 1)
    'mcherry_pos' : vector of cells that are mcherry positive (1)
    'mcherry_neg' : vector of cells that are mcherry negative (-1)
    
  path to save figures and data
    
::Outputs::
  'Day _ firing rates' : numpy file with a data matrix. Rows are cells, columns as follows:
    
    'mouse' : mouse number
    'exp' : mouse is control or experimental (0 or 1, respectively)
    'day' : sample day (-1 is habituation)
    'engram' : whether cell is mcherry positive, negative, or neither (1, -1, or 0)
    
    'no stim' : firing rate during no stim epochs
    'stim' : firing rate during stim epochs
    'second stim' : firing rate during second stim epoch
    'lick' : firing rate during licking
    'no lick' : firing rate during no licking
    'stim and lick' : firing rate during stimulation and licking
    'stim and no lick' : firing rate during stimulation and not licking
    'second stim and lick' : firing rate during second stimulation and licking
    'second stim and no lick' : firing rate during second stimulation and not licking
    'third no stim and lick' : firing rate during the third no stimulation and licking
    'third no stim and no lick' : firing rate during the third no stimulation and not licking

"""

#imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py
import pyabf
import general_utils as gen
import agg_plot_utils as agg

#file paths and save paths

file_path = r'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/data/cleaned habituation data/'
save_path = r'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/firing rate comparison/'

day = 1

save_figure = True
fr = 14.7985961923 # different for M223, later on in code
control_mice = ['M271','M275','M282','M283','M284','M285']

file_array = gen.find_hdf5(file_path)     

#%%

columns=['Mouse','Day','Experimental','mCherry', 'all rate', 'stim rate', 'no stim rate', 'post stim rate', 'second stim rate', 'third no stim rate', 'lick rate','no lick rate','stim lick rate','stim no lick rate', 'second stim lick rate','second stim no lick rate', 'third no stim lick rate', 'third no stim no lick rate']
numeric_columns = columns[4:]
agg_data = pd.DataFrame(columns = columns)

for file in file_array:

    print(file)
    
    filename = file.split('/')[-1]
    mouse = filename[:4]
    
    hdf5 = h5py.File(os.path.join(file_path, file),'r')
        
    # load in variables from hdf5 file
    time = np.array(hdf5["time"][()])
    spikes = np.array(hdf5["spikes"][()])
    stims_binary = np.array(hdf5["stims_binary"][()])
    licks = np.array(hdf5["licks"][()])
    mcherry_pos = np.array(hdf5["mcherry_pos"][()])
    mcherry_neg = np.array(hdf5["mcherry_neg"][()])
    
    cells = np.shape(spikes)[0]
    T = np.shape(spikes)[1]
    
    m223 = False
    if mouse == 'M223':
        fr = 56.653/2
        m223 = True
    
    name = 'Day '+str(day)+' '+mouse
    
    
    # find stim/nostim indices using stims_binary
    nostim_idx, stim_idx, poststim_idx, second_stim_idx, third_nostim_idx = agg.extract_stim_epochs(time, stims_binary,m223)
    
    # find lick/no lick indices using licks
    nolick_idx, lick_idx = agg.extract_lick_epochs(time, licks)
    
    # find stim and lick vs stim and no lick
    stim_lick_idx = np.intersect1d([stim_idx],[lick_idx])
    stim_nolick_idx = np.intersect1d([stim_idx],[nolick_idx])
    
    # find second stim and lick vs second stim and no lick
    second_stim_lick_idx = np.intersect1d([second_stim_idx],[lick_idx])
    second_stim_nolick_idx = np.intersect1d([second_stim_idx],[nolick_idx])
    
    #find third no stim with lick and no lick
    third_nostim_lick_idx = np.intersect1d([third_nostim_idx],[lick_idx])
    third_nostim_nolick_idx = np.intersect1d([third_nostim_idx],[nolick_idx])
    
    
    # calclulate the rates for each cell
    # all, stim, nostim, second stim, lick, no lick, stim and lick, stim and no lick, second stim and lick, second stim and no lick
    all_rate = []
    stim_rate = []
    nostim_rate = []
    poststim_rate = []
    second_stim_rate = []
    third_nostim_rate = []
    lick_rate = []
    nolick_rate = []
    stim_lick_rate = []
    stim_nolick_rate = []
    second_stim_lick_rate = []
    second_stim_nolick_rate = []
    third_nostim_lick_rate = []
    third_nostim_nolick_rate = []
    
    for i in range(cells):
        all_rate.append(np.sum(spikes[i,:])/len(spikes[0,:])*fr)
        stim_rate.append(np.sum(spikes[i,:][stim_idx])/len(stim_idx)*fr)
        nostim_rate.append(np.sum(spikes[i,:][nostim_idx])/len(nostim_idx)*fr)
        poststim_rate.append(np.sum(spikes[i,:][poststim_idx])/len(poststim_idx)*fr)
        second_stim_rate.append(np.sum(spikes[i,:][second_stim_idx])/len(second_stim_idx)*fr)
        third_nostim_rate.append(np.sum(spikes[i,:][third_nostim_idx])/len(third_nostim_idx)*fr)
        lick_rate.append(np.sum(spikes[i,:][lick_idx])/len(lick_idx)*fr)
        nolick_rate.append(np.sum(spikes[i,:][nolick_idx])/len(nolick_idx)*fr)
        stim_lick_rate.append(np.sum(spikes[i,:][stim_lick_idx])/len(stim_lick_idx)*fr)
        stim_nolick_rate.append(np.sum(spikes[i,:][stim_nolick_idx])/len(stim_nolick_idx)*fr)
        second_stim_lick_rate.append(np.sum(spikes[i,:][second_stim_lick_idx])/len(second_stim_lick_idx)*fr)
        second_stim_nolick_rate.append(np.sum(spikes[i,:][second_stim_nolick_idx])/len(second_stim_nolick_idx)*fr)
        third_nostim_lick_rate.append(np.sum(spikes[i,:][third_nostim_lick_idx])/len(third_nostim_lick_idx)*fr)
        third_nostim_nolick_rate.append(np.sum(spikes[i,:][third_nostim_nolick_idx])/len(third_nostim_nolick_idx)*fr)
    
    all_rate = np.array(all_rate)
    stim_rate = np.array(stim_rate) 
    nostim_rate = np.array(nostim_rate)
    poststim_rate = np.array(poststim_rate)
    second_stim_rate = np.array(second_stim_rate)
    third_nostim_rate = np.array(third_nostim_rate)
    lick_rate = np.array(lick_rate)
    nolick_rate = np.array(nolick_rate)
    stim_lick_rate = np.array(stim_lick_rate)
    stim_nolick_rate = np.array(stim_nolick_rate)
    second_stim_lick_rate = np.array(second_stim_lick_rate)
    second_stim_nolick_rate = np.array(second_stim_nolick_rate)
    third_nostim_lick_rate = np.array(third_nostim_lick_rate)
    third_nostim_nolick_rate = np.array(third_nostim_nolick_rate)
    

    sample = 'firing rate'

    plt.close('all')
    
    # aggregate the data
    # columns: mouse, exp?, day, engram?; no stim, stim, second stim, lick, no lick, stim and lick, stim and no lick, second stim and lick, second stim and no lick, third no stim and lick, third no stim and no lick
    
    days = np.full((1,cells),day)
    mouses = np.full((1,cells),mouse)
    if mouse in control_mice:
        exp = 0
    else:
        exp = 1
        
    exps = np.full((1,cells),exp)
    
    mcherry = mcherry_pos
    mcherry[np.where(mcherry_neg == -1)] = '-1'
    
    temp_data = np.vstack([mouses, days, exps, mcherry, all_rate, stim_rate, nostim_rate, poststim_rate, second_stim_rate, third_nostim_rate, lick_rate, nolick_rate, stim_lick_rate, stim_nolick_rate, second_stim_lick_rate, second_stim_nolick_rate, third_nostim_lick_rate, third_nostim_nolick_rate]).T
    mouse_data = pd.DataFrame(temp_data,columns=columns)
    
    for column in numeric_columns:
        mouse_data[column] = pd.to_numeric(mouse_data[column], errors='coerce')
    
    # create comp_data
    agg_data = pd.merge(agg_data, mouse_data, how='outer')


# save the data
name = 'Day '+str(day)+' Agg'
agg_data.to_pickle(os.path.join(os.path.join(save_path, 'data'), name+' firing rates'))

