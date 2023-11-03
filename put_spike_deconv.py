# -*- coding: utf-8 -*-
"""
Utility function to take the CaImAn output and extract putative spikes

@author: jnorm
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np


# applies threshold-mediated purging of spikes from CaImAn output data;
# returns a binary spike train (saves it to the data hdf5); path is location 
# on system of C, S, YrA, and data hdf5 files; noise_percentile and 
# spike_percentile are percentiles of the noise and spike probabilities below 
# which possible spike events will be purged
def caiman_spike_filter(S, C, YrA, dff, noise_percentile=50, spike_percentile=10, dff_threshold=30):
    
    # extract number of ROIs and trace length
    num_ROI = C.shape[0]
    trace_len = C.shape[1]
    
    # array that will hold filtered spikes
    spikes = np.zeros((num_ROI, trace_len), dtype=bool)
    
    # iterate through ROIs
    for ROI_idx in range(num_ROI):
        # extract trace, prepare mask of locations that exceed noise_percentile
        trace = C[ROI_idx,:]
        dff_trace = dff[ROI_idx,:]
        binary_mask = np.zeros(trace_len, dtype=bool)
        binary_mask2 = np.zeros(trace_len,dtype=bool)
        
        # convert noise_percentile to actual threshold
        if noise_percentile > 50: #using this gives much larger values, rarely go below 50 anyway
            one_hundred = np.percentile(np.absolute(YrA[ROI_idx,:]), 100)
            scale = noise_percentile/100
            threshold = one_hundred * scale
        else:
            threshold = np.percentile(np.absolute(YrA[ROI_idx,:]), noise_percentile)
            
        # convert dff_threshold to percentage
        dff_threshold_percent = dff_threshold/100
            
        # threshold by noise 
        for i, val in enumerate(trace):
            if val >= threshold:
                binary_mask[i] = 1
        
        # threshold by dff value
        for i, val in enumerate(dff_trace):
            if val >= dff_threshold_percent:
                binary_mask2[i] = 1
        dff_above = np.sum(binary_mask2)/trace_len
        
        # determine threshold for spike probabilities (takes as percentile of
        # nonzero spike probabilities)
        non_zero_idx = np.nonzero(S[ROI_idx,:])
        spike_threshold = np.percentile(S[ROI_idx,:][non_zero_idx], spike_percentile)
        #spike_threshold = np.percentile(S[ROI_idx,:], spike_percentile)
        if dff_above > .03: #this was tested by trial and error
            spike_threshold = spike_threshold*.03/dff_above
        
        # combine with previous thresholding to produce spike train
        for i, val in enumerate(S[ROI_idx,:]):
            if val >= spike_threshold and binary_mask[i] == 1 and binary_mask2[i] == 1:
                spikes[ROI_idx, i] = 1
        
    return spikes

