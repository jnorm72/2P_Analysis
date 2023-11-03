# -*- coding: utf-8 -*-
"""
Functions to support the data cleaning script

@author: jnorm
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import h5py
import pyabf
import put_spike_deconv as jv


# functions        
        
def trace_visualization_spikes(traces1, spikes, time, start_idx=0, end_idx=50, fr=15, save_figures=False, savePath='', imgName='traces and spikes', title=''):
    #plots the traces with spikes
    """

    Parameters
    ----------
    traces1 : an array with dimensions time x cells that you want to plot
    
    spikes : an array with the same dimensions as traces1, but with binarized spikes included
    
    start_idx : start index for plotting. Can't be above end_idx
        DESCRIPTION. The default is 0.
        
    end_idx : end index for plotting. Can't be below start_idx
        DESCRIPTION. The default is 50.
        
    save_figures : whether figures should be saved or not
       
    save_Path : string. to the location you want the figures saved if desired. make sure it ends in '/'
    
    imgName : string. will be the saved name of the image
    
    title : string. will be the title of the figure


    """
    
    T = traces1.shape[1]
    numcells1 = np.shape(traces1)[0]
    #T1 = np.shape(traces1)[1]
    excitatory_color = 'green'
    spike_color = 'red'
    
    #separated is with the traces separated apart for visualization
    traces1_separated = traces1+np.reshape(np.repeat(np.arange(numcells1),T), (numcells1, T))
    spikes_separated = spikes+np.reshape(np.repeat(np.arange(numcells1),T), (numcells1, T))
    
    #plot raw traces
    trace_fig1 = plt.figure()
    trace_ax1 = plt.axes()
    
    x = np.arange(numcells1)
    x = x[1:]
    labels = np.arange(T)
    for i in x:
        trace_ax1.plot(time,traces1_separated[i,:], color=excitatory_color, linewidth=.2) 
        trace_ax1.plot(time,spikes_separated[i,:], color=spike_color, linewidth=.2) 
    
    plt.ylim((start_idx, end_idx))
    plt.yticks(np.arange(start_idx, end_idx,5))
    plt.xlim(0,max(time))
    start, end = trace_ax1.get_xlim()
    plt.xticks(np.arange(0,int(max(time)),30), fontsize=8)
    plt.xlabel('Time (seconds)',fontsize=14)
    plt.ylabel('Cell #', fontsize=14)
    plt.title(title,fontsize=20)
    if save_figures:
        trace_fig1.savefig(savePath+imgName+'.jpg',dpi=400)
        

def hdf5_cell_cleaner(hdf5, pop_idx, is_temp=True, list_of_keys=['A','traces','C','S','spikes','YrA','time','stims','stims_binary','licks']):
    #update the variables if yours are different, these were the ones that I saved
    """
    Parameters
    ----------
    hdf5 : hdf5 file with variables saved and in r+ mode
        
    pop_idx : array of indices that should be popped from the hdf5 file. traces start at 1. 
    
    """
    keys = hdf5.keys()
    
    for key in list_of_keys:
        
        #check that the given key is in the hdf5 file
        if key not in keys:
            print(key)
            raise Exception("given key not in estimates file")
            
        #open data in temp variable
        temp = hdf5[key]
        
        # delete the desired indices, but only if the variable has 2 dimensions
        if temp.ndim == 2:
            #A has pixels by cells, so go the other way
            if key == 'A':
                temp = np.delete(temp, pop_idx, 1)
            else:
                temp = np.delete(temp, pop_idx, 0)
           
        #save the changes, either permanently or temporarily
        if not is_temp:
            del hdf5[key]
            temp2 = hdf5.create_dataset(key, data=temp)
        if is_temp:
            if key+'2' in keys:
                del hdf5[key+'2']
            temp2 = hdf5.create_dataset(key+'2', data=temp)
        

def hdf5_time_cleaner(hdf5, excise_start_time, excise_end_time, is_temp=True, list_of_keys=['traces','C','S','spikes','YrA','time','stims','stims_binary','licks']):
    #update the variables if yours are different, these were the ones that I saved
    """
    Parameters
    ----------
    hdf5 : hdf5 file with variables saved and in r+ mode. make sure there is a 'time' key

    excise_start_time : float
        start time for when data should be excised
        
    excise_end_time : float
        end time for when data should be excised

    """
    keys = hdf5.keys()

    #convert times to indices    
    excise_start_idx = np.min(np.where(hdf5['time'][()]>excise_start_time))
    excise_end_idx = np.min(np.where(hdf5['time'][()]>excise_end_time))

    
    #remove the times for all variables
    for key in list_of_keys:
        
        #check that the given key is in the hdf5 file
        if key not in keys:
            print(key)
            raise Exception("given key not in estimates file")
        #open data in temp variable
        temp = hdf5[key]
        
        # delete the desired times from all variables
        if temp.ndim == 2:
            temp = np.delete(temp, np.arange(excise_start_idx,excise_end_idx), 1)
        else:
            temp = np.delete(temp, np.arange(excise_start_idx,excise_end_idx), 0)


        #save the new variable, either permanently or temporarily
        if not is_temp:
            #delete the old variable
            del hdf5[key]
            # save the new variable in the original hdf5 file
            temp2 = hdf5.create_dataset(key,data=temp)
        
        if is_temp:
            if key+'2' in keys:
                del hdf5[key+'2']
            temp2 = hdf5.create_dataset(key+'2',data=temp)
            

    
def hdf5_abf_concat(hdf5, abf):
    #used to add licks into the hdf5 file - can be edited to add other stimuli if needed
    
    # load the hdf5 file, pull the time variable
    time_hdf5 = hdf5['time']
    len_img = len(time_hdf5)
    
    # load the abf file
    abf.setSweep(sweepNumber=0, channel=0)
    time_abf = abf.sweepX
    stims = abf.sweepY
    
    #use the jv utils to have a binary mask of the stimulation epochs. stims_norm is returned but not used
    stims_binary, stims_norm = jv.extract_epochs(abf, 0)
    stims_binary = np.array(stims_binary)
    
    abf.setSweep(sweepNumber=0, channel=1)
    licks = abf.sweepY
    
    #extend the licks before downsampling
    licks = extend_licks(licks, time_abf)
    
    #find the initial stim time
    start_stim = np.min(np.where(stims>.75))
    print(start_stim)
    
    #remove the time from the abf before the imaging starts
    initial_time_abf = start_stim-180000
    final_time_abf = np.around(time_hdf5[-1],3)*1000+initial_time_abf
    final_time_abf = int(final_time_abf)
    
    time_abf = time_abf[initial_time_abf:final_time_abf]-initial_time_abf/1000
    stims = stims[initial_time_abf:final_time_abf]
    licks = licks[initial_time_abf:final_time_abf]
    stims_binary = stims_binary[initial_time_abf:final_time_abf]
    print(stims_binary)
    
    #round the both sets of times to nearest thousandths
    time_abf = np.around(time_abf,3)
    time_hdf5 = np.around(time_hdf5,3)
    
    #find the indices in abf_time that matches the imaging times
    abf_included = np.in1d(time_abf,time_hdf5)
    print(len(abf_included))
    
    #filter the abf files by the boolean array
    stims_img = stims[abf_included]
    licks_img = licks[abf_included]
    stims_binary_img = stims_binary[abf_included]
    
    #insert a 0 at the beginning of the array so the lengths match
    stims_img = np.insert(stims_img,0,0)
    licks_img = np.insert(licks_img,0,0)
    stims_binary_img = np.insert(stims_binary_img,0,0)
    
    print(len(licks_img))
    time_img = time_abf[abf_included]
    time_img = np.insert(time_img,0,0)
    
    #save the stims and licks into the hdf5
    stims_final = hdf5.create_dataset('stims',data=stims_img)
    licks_final = hdf5.create_dataset('licks',data=licks_img)
    stims_binary_final = hdf5.create_dataset('stims_binary',data=stims_binary_img)
    
    

def extend_licks(licks, time):
    #script to account for the delay period after a mouse licks
    #make sure that this is before downsampling
    # used in hdf5_abf_concat
    """
    Parameters
    ----------
    licks : array
        should be an array containing licks where the licks are above 2 and the baseline is below 2
    time : array
        should be an array containing the time with the same dimensions as licks

    Returns
    -------
    licks : array
        array with licks extended by half a second

    """

    # if 'licks' not in hdf5.keys():
    #     raise Exception("no licks in hdf5")
    # licks = hdf5['licks']
    # licks_array = np.array(licks)
    
    #pull the frame rate from time
    fr = len(time)/time[-1]
    ext_per = int(fr/2) #change this if the desired extension length changes from .5 seconds
    
    #threshold the licks
    licks[licks <= 2] = 0
    licks[licks > 2] = 1
    #find where licks is 1, create a boolean array
    temp = np.in1d(licks,[1])
    #loop through the array and find where there are licks, then extend them
    for ind in range(len(temp)-ext_per):
        if temp[ind:ind+ext_per].any():
            licks[ind+ext_per] = 1
    
    return licks
    

        

def spike_adder(hdf5, noise_percentile=50, spike_percentile=10, dff_threshold=30):
    #uses put_spike_deconv to calculate the spikes from the parameters in the files
    
    S = hdf5['S']
    C = hdf5['C']
    YrA = hdf5['YrA']
    traces = hdf5['traces']
    
    spikes = jv.caiman_spike_filter(S, C, YrA, traces, noise_percentile, spike_percentile, dff_threshold)
    
    if 'spikes' in hdf5.keys():
        del hdf5['spikes']
    temp = hdf5.create_dataset('spikes',data=spikes)
    hdf5['spikes'].attrs["noise_percentile"] = noise_percentile
    hdf5['spikes'].attrs["spike_percentile"] = spike_percentile
    hdf5['spikes'].attrs["dff_threshold"] = dff_threshold
    

    
    
    
    