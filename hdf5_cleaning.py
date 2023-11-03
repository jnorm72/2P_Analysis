# -*- coding: utf-8 -*-
"""
Pipeline for cleaning the data - can remove cells and time points that are noisy

@author: jnorm
"""

# script expects hdf5 files, and will return hdf5 files with the updated variables
# general structure:
# Code block 1 - load in the file
# Code block 2 - view the desired cells, write the desired edits in Cell 3
# Code block 3 - view the changes
# Code block 4 - save the changes (can't undo after this step)

import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pyabf
import cleaning_utils as cln
import general_utils as gen


#set up file path and load hdf5
file_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/data/cleaned day 1 data/'
save_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/behavior cleaning/' #only for saving raw and cleaned images

i = 0 #increase this to look through files in a folder

def behavior_cleaning(hdf5, cell_cleaner, time_cleaner, start_idx, end_idx, pop_idx, excise_start_time, excise_end_time, is_temp=True):
    if cell_cleaner and time_cleaner:
        raise Exception('You cannnot clean both at the same time, do one or the other')
    
    if cell_cleaner:
        cln.hdf5_cell_cleaner(hdf5, pop_idx, is_temp=is_temp)
            
        #plot the cleaned data with spikes
        spikes2 = hdf5["spikes2"]
        f_dff2 = hdf5["traces2"]
        time2 = hdf5['time2']
        cln.trace_visualization_spikes(f_dff2, spikes2, time2, start_idx=start_idx,end_idx=end_idx, save_figures=True, savePath=save_path, imgName=name+' cleaned', title=name+' cleaned')
    
    #clean the motion times
    if time_cleaner:
        cln.hdf5_time_cleaner(hdf5, excise_start_time, excise_end_time, is_temp=is_temp)
        
        #plot the cleaned data with spikes
        spikes2 = hdf5["spikes2"]
        f_dff2 = hdf5["traces2"]
        time2 = hdf5['time2']
        cln.trace_visualization_spikes(f_dff2, spikes2, time2, start_idx=start_idx,end_idx=end_idx, save_figures=True, savePath=save_path, imgName=name+' cleaned', title=name+' cleaned')
    
file_array = gen.find_hdf5(file_path)
file = file_array

# this is only necessary if adding behavior into the hdf5 file
abf_file_array = gen.find_abf(file_path)
abf_file = abf_file_array

filename = file[i].split('/')[-1]
mouse = filename[1:5]
date = filename[5:-10]

# check to make sure that mouse has an abf file in the folder - this might not be necessary for test experiments
for abf_f in abf_file: 
    if mouse in abf_f: 
        abf_name = abf_f
        print('abf file found')

name = mouse+' '+date

#%%
hdf5 = h5py.File(os.path.join(file_path, filename),'r+')
abf = pyabf.ABF(abf_name)

#spike adder - important to add the spikes
cln.spike_adder(hdf5)

#hdf5_concat - add in stims, stims_binary, and licks to hdf5 file
if 'licks' not in hdf5.keys():
    cln.hdf5_abf_concat(hdf5, abf)
  
#get variables from the files
f_dff = hdf5["traces"]
spikes = hdf5["spikes"]
time = hdf5['time']

print(np.shape(f_dff[()])[0])

#plot the file with spikes - update these indices as you look through the cells
start_idx = 0
end_idx = 50
cln.trace_visualization_spikes(f_dff, spikes, time, start_idx,end_idx, save_figures=True, savePath=save_path, imgName=name+' raw', title=name+' raw')

#put a stop in after the plot
#%%
cell_cleaner = False #can only set one of these to true at a time - can't clean both the cells and the time simultaneously
time_cleaner = True

pop_idx = [26,27] #cells to get rid of
#pop_idx = np.arange(55,92) #cells to get rid of, but removes a whole block of cells
if cell_cleaner:
    print(end_idx-start_idx-len(pop_idx))
excise_start_time = 238 #time to start removing activity data
excise_end_time = 250 # time to end removing activity data

behavior_cleaning(hdf5, cell_cleaner, time_cleaner, start_idx, end_idx, pop_idx, excise_start_time, excise_end_time, is_temp=True)   

#%%     
#save changes
if cell_cleaner:
    cln.hdf5_cell_cleaner(hdf5, pop_idx, is_temp=False)
if time_cleaner:
    cln.hdf5_time_cleaner(hdf5, excise_start_time, excise_end_time, is_temp=False)
hdf5.close()
plt.close('all')







