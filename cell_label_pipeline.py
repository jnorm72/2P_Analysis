# -*- coding: utf-8 -*-
"""
Identifies labeled, unlabeled, and label uncertain cells

@author: jnorm
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import pyabf
import pickle
import general_utils as gen
import cell_label_utils as cel


#set up file path and load hdf5 - should be customized to your folder formats
hdf5_file_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/data/cleaned Day 3 data/' #only green channel of the trial
save_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/mcherry identification/' # these folders are hard coded in
red_file_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/data/red image data/' #paths might be hard coded as well?
tif_file_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/data/correlation image data/'

show_figure = False

#find the hdf5 files and print them
file_array = gen.find_hdf5(hdf5_file_path)
files = file_array
print(file_array)



#loop through each of the hdf5 files
for i, file in enumerate(files):

    print(file)
    
    # split the file names
    filename = file.split('/')[-1]
    mouse = filename[:4]
    date = filename[5:-10]
     
    name = mouse+' '+date
    
    hdf5 = h5py.File(os.path.join(hdf5_file_path, filename),'r+')
    
    # check if there are red images for the mouse
    red_file_path_full = os.path.join(os.path.join(red_file_path, mouse),date) #red file path is the red_file_path plus the mouse, then the date
    is_red = cel.check_red(red_file_path_full)
    print(is_red)
    
    if not is_red:
        # if there isn't a red file, mark everything as zero and move on. This saves to the hdf5 file
        cel.create_zeros(hdf5)
    
    else:    
        # z project all of the red images, both red and green channels
        A_green_zproject, A_red_zproject = cel.z_project(red_file_path_full)
        #%%
        # Loading in the hdf5 ROIs - rois is 3D with each ROI being its own image, while roi_sum is a 2D version of the rois
        rois, roi_sum = cel.load_hdf5_roi(hdf5)
        
        # Loading in the Cn matrix
        #loads in the correlation image from a folder containing all correlation images
        #requires the mouse and date naming strategy
        cn_matrix = cel.load_correlation_image(tif_file_path, mouse, date)
        
        # find the medians for the red and green images
        #takes in the numpy arrays and returns numpy arrays
        A_green_median, A_red_median = cel.red_image_process(A_green_zproject, A_red_zproject)
        
        # align the two green images
        offset = cel.whole_image_find_offset(cn_matrix, A_green_median, dimensionality=0.9)
        
        
        #shifts the z projected numpy arrays by the specified offsets
        A_green_adjusted, A_red_adjusted = cel.adjust_image(A_green_zproject, A_red_zproject, offset)
        
        # plot and save overlay
        cel.visualize_offset(A_green_median, A_green_adjusted, cn_matrix, roi_sum, offset, name, save_path, show_figure) #has the "motion correction" folder hard coded in
        
        
        #%%
        # subtract out green red image from red red image
        
        bleed_through_ratio = .25
        
        #subtracts the green image from the red image to eliminate cells that are bright only due to bleed-through
        A_green_adjusted_median, A_red_adjusted_median, subtracted_image = cel.subtract_red_from_green(A_green_adjusted, A_red_adjusted, bleed_through_ratio)
        
        #used for visualizing the subtraction, again folder path is hard coded in
        cel.visualize_red_minus_green(A_green_adjusted_median, A_red_adjusted_median, subtracted_image, name, save_path, show_figure)
        
        #%%
        
        std_threshold = .45 #how much brighter the cell needs to be than the pixed mean of the surrounding area
        frame_window = 100 #how big of a surrounding area to sample from (in pixels) to determine the local background
        
        #find the mcherry positve cells - inputs and outputs detailed in the function
        mcherry_pos, mcherry_pos_dict, num_positive = cel.select_mcherry_positive_frame_window(subtracted_image, rois, std_threshold, frame_window)
        
        #Finding the non-red cells - only necessary because mCherry was very dim, not enough illumination across the FOV 
        # find centroids of all rois
        cell_idx_list = cel.find_cell_idx(rois)
        
        # find peaks for mcherry positive ones
        mcherry_pos_idx, mcherry_idx_list = cel.find_mcherry_idx(cell_idx_list, mcherry_pos_dict)
        
        cel.visualize_positive_mcherry_cells(mcherry_pos, subtracted_image, roi_sum, False, name, save_path, show_figure)
        
        # identify mcherry negative cells
        mcherry_neg_idx, mcherry_neg = cel.select_mcherry_negative(rois, cell_idx_list, mcherry_pos_idx, mcherry_idx_list)
        
        print(num_positive)
        print(num_positive/len(mcherry_neg_idx))
        
        # plot the positive, negative, and both (hard coded file paths)
        cel.visualize_positive_mcherry_cells(mcherry_neg, subtracted_image, roi_sum, True, name, save_path, show_figure)
        
        cel.visualize_positive_negative_mcherry_cells(mcherry_pos, mcherry_neg, subtracted_image,name, save_path, show_figure)
        
        #%%
        # save the mcherry positive and negative cells in the hdf5 file
        cel.save_mcherry_hdf5(hdf5, mcherry_pos_idx, mcherry_neg_idx)
        






