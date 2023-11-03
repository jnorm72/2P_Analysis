# -*- coding: utf-8 -*-
"""
Functions to help identify labelled cells

@author: jnorm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.spatial as sp
import os
import h5py
import find_offset
import general_utils as gen


# functions

def check_red(red_file_path_full):
    #simply checks if the path exists - some mice had poor quality so no red image was even taken
    is_red = os.path.exists(red_file_path_full)
    
    return is_red

def create_zeros(hdf5):
    # delete if prior data
    keys = ['tdtm', 'mcherry_pos','mcherry_neg']
    for key in keys:
        if key in hdf5.keys():
            del hdf5[key]
    
    temp_pos = np.zeros(np.shape(hdf5['traces'][()])[0])
    temp_neg = np.zeros(np.shape(hdf5['traces'][()])[0])
    
    temp1 = hdf5.create_dataset('mcherry_pos' ,data=temp_pos)
    temp2 = hdf5.create_dataset('mcherry_neg' ,data=temp_neg)

def z_project(filepath):
    #z project all the raw files in a folder
    
    folders = [f for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]
    
    i = 1
    
    # Loading in the red image matrices
    A_green, A_red = gen.load_raw(os.path.join(filepath, folders[0]))
    
    for folder in folders[1:]:
        A_green_temp, A_red_temp = gen.load_raw(os.path.join(filepath, folder))
        A_green = A_green + A_green_temp
        A_red = A_red + A_red_temp
        i = i+1
    
    A_green_zproject = A_green/i
    A_red_zproject = A_red/i
    
    return A_green_zproject, A_red_zproject


def load_hdf5_roi(hdf5):
    # Loading in the hdf5 ROIs - rois is 3D with each ROI being its own image, while roi_sum is a 2D version of the rois
    
    #check that A is in the hdf5 file
    if 'A' not in hdf5.keys():
            raise Exception("given key not in estimates file")
            
    A = hdf5['A'][()]
    num_pixels = np.sqrt(A.shape[0]).astype(int)
    
    rois = np.reshape(A, (num_pixels, num_pixels, np.shape(A)[1]))*1000
    rois = np.transpose(rois, (1,0,2))
    rois[rois<10] = 0
    
    roi_sum = np.sum(rois, axis=2)
    roi_sum = np.reshape(roi_sum,(num_pixels, num_pixels))*1000
    roi_sum[roi_sum<10] = 0
    
    return rois, roi_sum

def load_correlation_image(file_path, mouse, date):
    #loads in the correlation image from a folder containing all correlation images
    #requires the mouse and date naming strategy
    
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename) 
        if ((mouse in f) and (date in f) and ('lick' in f)):
            cn = gen.load_tif(os.path.join(file_path, f))
    return cn

def red_image_process(A_green, A_red):
    """
    Finds medians of the red and green channels of the red image.

    ::Inputs::
        A_green :numpy array: - Green channel of the red image.
        A_red   :numpy array: - Red channel of the red image.

    ::Outputs::
        A_green_median :numpy array: - Median image of green channel of red image
        A_red_median   :numpy array: - Median image of red channel of red image
    """

    # Finding the median of the red matrix data
    A_green_median = np.median(A_green, 0)
    A_red_median = np.median(A_red, 0)

    return A_green_median, A_red_median

def whole_image_find_offset(cn_matrix, A_green_median, dimensionality=0.9):
    """
    Finds the offset with the entire image as opposed to the fundtion above.

    ::Inputs::
        cn_matrix       :numpy array: - cn image
        A_green_median  :numpy array: - Median image of green channel of red image
        dimensionality  :float: - see 'Danny_find_offset.py'

    ::Outputs::
        offset :numpy array: - offset of A_green_median with respect to cn_matrix
    """

    # Zscoreing
    fov_1 = stats.zscore(A_green_median, axis=None)
    fov_2 = stats.zscore(cn_matrix, axis=None)

    # Calling the offset function - returns the pixel shifts required
    offset = find_offset(fov_1, fov_2, fov_1.shape, dimensionality)

    return offset

def adjust_image(A_green, A_red, offset):
    """
    Adjusting the images so that they line up accordingly

    ::Inputs::
        A_green :numpy array: - Green channel of the red image.
        A_red   :numpy array: - Red channel of the red image.
        offset  :numpy array: - offset of A_green_median with respect to cn_matrix

    ::Outputs::
        A_green_adjusted :numpy array: - Adjusted green channel of the red image
        A_red_adjusted   :numpy array: - Adjusted red channel of the red image
    """

    # Init adusted frames
    A_green_adjusted = A_green.copy()
    A_red_adjusted = A_red.copy()
    
    green_min = np.min(A_green)
    red_min = np.min(A_red)

    # Looping through all of the frames
    for j, frame in enumerate(A_green):

        # Init temp matricies
        temp_g = A_green[j,:,:]
        temp_r = A_red[j,:,:]
        
        # Pushing the cols over
        # Adding rows of 0s to the right side of the image
        if offset[0] > 0:
            for i in range(abs(offset[0])):
                # Green image
                temp_g = np.concatenate((np.ones((len(temp_g), 1))*green_min, temp_g), axis=1) # Adds new column to the front
                temp_g = np.delete(temp_g, -1, axis=1) # Delete the extra columns

                # Red image
                temp_r = np.concatenate((np.ones((len(temp_r), 1))*red_min, temp_r), axis=1) # Adds new column to the front
                temp_r = np.delete(temp_r, -1, axis=1) # Delete the extra columns

        # Adding rows of zeros to the left side of the image
        elif offset[0] < 0:
            for i in range(abs(offset[0])):
                # Green Image
                temp_g = np.concatenate((temp_g, np.ones((len(temp_g), 1))*green_min), axis=1) # Adds new column to the back
                temp_g = np.delete(temp_g, 0, axis=1) # Delete the extra columns

                # Red Image
                temp_r = np.concatenate((temp_r, np.ones((len(temp_r), 1))*red_min), axis=1) # Adds new column to the back
                temp_r = np.delete(temp_r, 0, axis=1) # Delete the extra columns

        else:
            pass

        # Pushing rows over
        # Adds columns of zeros to right side of the image
        if offset[1] > 0:
            for i in range(abs(offset[1])):
                # Green Image
                temp_g = np.concatenate((temp_g, np.ones((1, len(temp_g)))*green_min), axis=0) # Adds row
                temp_g = np.delete(temp_g, 0, axis=0) # deletes extra rows

                # Red Image
                temp_r = np.concatenate((temp_r, np.ones((1, len(temp_r)))*red_min), axis=0) # Adds row
                temp_r = np.delete(temp_r, 0, axis=0) # deletes extra rows

        # Adds columns to the left side of the image
        elif offset[1] < 0:
            for i in range(abs(offset[1])):
                # Green Image
                temp_g = np.concatenate((np.ones((1, len(temp_g)))*green_min, temp_g), axis=0) # Adds row
                temp_g = np.delete(temp_g, -1, axis=0) # Deletes extra row 

                # Red Image
                temp_r = np.concatenate((np.ones((1, len(temp_r)))*red_min, temp_r), axis=0) # Adds row
                temp_r = np.delete(temp_r, -1, axis=0) # Deletes extra row            

        else:
            pass

        # Adding new arrays
        A_green_adjusted[j,:,:] = temp_g
        A_red_adjusted[j,:,:] = temp_r

    return A_green_adjusted, A_red_adjusted

def visualize_offset(A_green_median, A_green_adjusted, cn_matrix, rois, offset, name, save_path, show_figure):
    #the save path has "motion correction" folder hard coded in
    """
    Visualizing how the offset program works. Good to check to see if your offset is actually working correctly.

    ::Inputs::
        A_green_median       :numpy array: - median image of the green channel of the red image
        A_green_adjusted     :numpy array: - Adjusted green channel of the red image
        cn_matrix            :numpy array: - cn image
        rois                 :numpy array: - Image with all the ROIs
    """
    plt.close()

    # Finding the median of the adjusted image
    A_green_adjusted_median = np.median(A_green_adjusted, 0)

    # Adding a copy of A_green_median to reset the brightness everytime
    A_green_median_original = np.copy(A_green_median)
    A_green_adjusted_median_original = np.copy(A_green_adjusted_median)


    # Masking the ROI image to make 0 see-through
    masked_roi = np.ma.masked_where(rois == 0, rois)

    alpha = .5
    fontsize = 9

    # Plotting the non adjusted image
    plt.subplot(1,3,1)
    plt.imshow(A_green_median, cmap="gray")
    plt.imshow(masked_roi, alpha=alpha, cmap="hot")
    plt.axis("off")
    plt.title("Before offset", fontsize=fontsize)


    # Plotting the adjusted image
    plt.subplot(1,3,2)
    plt.imshow(A_green_adjusted_median, cmap="gray")
    plt.imshow(masked_roi, alpha=alpha, cmap="hot")
    plt.axis("off")
    plt.title("After offset", fontsize=fontsize)

    # Plotting the CN image
    plt.subplot(1,3,3)
    plt.imshow(cn_matrix, cmap="gray")
    plt.imshow(masked_roi, alpha=alpha, cmap="hot")
    plt.axis("off")
    plt.title("CN Image", fontsize=fontsize)


    if show_figure:
        plt.show()
    plt.savefig(os.path.join(os.path.join(save_path, 'motion correction'), name+' offset visualization'+'.png'),dpi=400, bbox_inches='tight')
        
def subtract_red_from_green(A_green_adjusted, A_red_adjusted, bleed_through_ratio):
    """    
    Finding the median and subtracting the green out from the red image

    ::Inputs::
        A_green_adjusted :numpy array: - Adjusted green channel of the red image
        A_red_adjusted   :numpy array: - Adjusted red channel of the red image
        bleed_through_ratio :float: - the ratio (found by linear regression) to use when finding the subtracted image

    ::Outputs::
        A_green_adjusted_median :numpy array: - Median of all frames of the green channel of the red image
        A_red_adjusted_median :numpy array: - Median of all frames of the red channel of the red image
        subtracted_image :numpy array: - Image after subtracting A_green_adjusted_median from A_red_adjusted_median

    """
    A_green_adjusted_median = np.median(A_green_adjusted, 0)
    A_red_adjusted_median = np.median(A_red_adjusted, 0)
    subtracted_image = A_red_adjusted_median - A_green_adjusted_median * bleed_through_ratio

    return A_green_adjusted_median, A_red_adjusted_median, subtracted_image

def visualize_red_minus_green(A_green_adjusted_median, A_red_adjusted_median, subtracted_image, name, save_path, show_figure):
    """
    Visualization of the red image minus the green image

    ::Inputs::
        A_green_adjusted_median :numpy array: - Median of all frames of the green channel of the red image
        A_red_adjusted_median :numpy array: - Median of all frames of the red channel of the red image
        subtracted_image :numpy array: - Image after subtracting A_green_adjusted_median from A_red_adjusted_median
    """
    plt.close()
    
    fontsize = 9
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(A_green_adjusted_median, cmap="gray")
    plt.axis('off')
    plt.title("Green Median Image", fontsize=fontsize)
    
    plt.subplot(1,3,2)
    plt.imshow(A_red_adjusted_median, cmap="gray")
    plt.axis('off')
    plt.title("Red Median Image", fontsize=fontsize)
    
    plt.subplot(1,3,3)
    plt.imshow(subtracted_image, cmap="gray")
    plt.axis('off')
    plt.title("Subtracted Red Image", fontsize=fontsize)
    
    if show_figure:
        plt.show()
    plt.savefig(os.path.join(os.path.join(save_path, 'red green subtraction'), name+' red green subtraction'+'.png'),dpi=400, bbox_inches='tight')

def select_mcherry_positive_frame_window(subtracted_image, A_array, std_threshold, frame_window):
    """
    Attempting to find mcherry positive cells with a window around each ROI

    ::Inputs::
        subtracted_image :numpy array: - Image after subtracting A_green_adjusted_median from A_red_adjusted_median
        A_array :numpy array: - numpy array version of A_csc
        std_threshold :float: - How many standard deviations away from the mean to confirm that a cell is mcherry positive
        frame_window :int: - The pixel window around each ROI to take into account

    ::Outputs::
        mcherry_pos :numpy array: - Image of all the positive mCherry cells
        mcherry_pos_dict :dict: - Dictionary of all the positive mCherry cells
        num_positive :int: - The number of positive mcherry cells
    """
    dims = np.shape(A_array)[0]
    # Init list of mcherry positive neruons
    mcherry_pos = np.zeros((dims, dims))
    mcherry_pos_dict = {}
    num_positive = 0

    # Looping through all of the ROIs
    for i, roi in enumerate(A_array[0, 0, :]):
        
        
        A_roi = A_array[:,:,i]
        # Finding the max value of the roi
        max_roi = np.where(A_roi == A_roi.max())

        # Picking one location if there is more than two with the max value
        if len(max_roi[0]) > 1:
            x_coord = max_roi[0][0]
            y_coord = max_roi[1][0]
        else:
            x_coord = max_roi[0]
            y_coord = max_roi[1]

        # Creating the variables that describe the window around the ROI and makes it non-zero and
        # not bigger than the max values of the frame
        x1 = max(0, int(x_coord - frame_window/2))
        x1 = min(x1, dims)
        x2 = max(0, int(x_coord + frame_window/2))
        x2 = min(x2, dims)
        y1 = max(0, int(y_coord - frame_window/2))
        y1 = min(y1, dims)
        y2 = max(0, int(y_coord + frame_window/2))
        y2 = min(y2, dims)

        # Making a boolean matrix where the ROI is
        bool_A_reshape = A_roi > 0

        # Making a boolean matrix where ROI isn't
        bool_A_reshape_inverse = np.copy(A_roi)
        where_0 = np.where(bool_A_reshape == 0)
        where_1 = np.where(bool_A_reshape == 1)
        bool_A_reshape_inverse[where_0] = 1
        bool_A_reshape_inverse[where_1] = 0

        # Finding the mean brightness of the roi
        one_neuron_fluorescence_mean = np.sum(np.multiply(A_roi[x1:x2, y1:y2], subtracted_image[x1:x2, y1:y2])) / np.sum(A_roi[x1:x2, y1:y2])

        # Finding the mean brightness and standard deviation around the roi
        background_fluorescence_mean = np.sum(np.multiply(bool_A_reshape_inverse[x1:x2, y1:y2], subtracted_image[x1:x2, y1:y2])) / np.sum(bool_A_reshape_inverse[x1:x2, y1:y2])
        background_fluorescence_std = np.sqrt(np.sum((np.multiply(bool_A_reshape_inverse[x1:x2, y1:y2], subtracted_image[x1:x2, y1:y2]) - background_fluorescence_mean) ** 2) / np.sum(bool_A_reshape_inverse[x1:x2, y1:y2]))

        # Comparing the two values
        if one_neuron_fluorescence_mean > background_fluorescence_mean + std_threshold * background_fluorescence_std:
            mcherry_pos += A_roi
            mcherry_pos_dict[i] = A_roi
            num_positive += 1 
        
    return mcherry_pos, mcherry_pos_dict, num_positive     

    
def find_cell_idx(rois):
    #takes in an array of ROIs, returns a list of the indices for the peak value for each ROI
    
    cell_idx_list = np.zeros((1,2))
    for i, roi in enumerate(rois[0,0,:]):
       temp_roi = rois[:,:,i]
       maxs = np.where(temp_roi == np.max(temp_roi))
       cent = (np.mean(maxs[0]),np.mean(maxs[1]))
       cent = np.around(cent,0)
       cell_idx_list = np.vstack((cell_idx_list, cent))
    
    cell_idx_list = cell_idx_list[1:,:]
    
    return cell_idx_list
    
    
def find_mcherry_idx(cell_idx_list, mcherry_pos_dict):
    #takes in an array of cell indices and the dictionary of mcherry positive neurons, returns an array of mcherry positve indices
    mcherry_pos_idx = []
    for key in mcherry_pos_dict.keys():
        mcherry_pos_idx.append(key)
        
    mcherry_idx_list = cell_idx_list[mcherry_pos_idx]
    
    return mcherry_pos_idx, mcherry_idx_list
        

def visualize_positive_mcherry_cells(mcherry_pos, subtracted_image, A_reshape_tot_neuron, neg, name, save_path, show_figure):
    """
    Plots the subtracted image, the subtracted image with an overlay of the mCherry positive cells,
    and the subtracted image with an overlay of all the ROIs

    ::Inputs::
        mcherry_pos :numpy array: - Image of all the positive mCherry cells
        subtracted_image :numpy array: - Image after subtracting A_green_adjusted_median from A_red_adjusted_median
        A_reshape_tot_neuron :numpy array: - Image with all the ROIs
        neg : - Boolean whether the sample is negative or not
    """
    plt.close()

    alpha = .5
    fontsize = 9
    
    # set up positive and negative toggle
    color = 'hot'
    sample = 'positive'
    if neg:
        color = 'winter'
        sample = 'negative'
        

    # Plotting the subtracted image
    plt.subplot(1,3,1)
    plt.imshow(subtracted_image, cmap="gray")
    plt.axis('off')
    plt.title("Subtracted Image", fontsize=fontsize)

    # Plotting the positive cells
    # Creating a mask so the image works better
    masked_roi = np.ma.masked_where(mcherry_pos == 0, mcherry_pos)
    plt.subplot(1,3,2)
    plt.imshow(subtracted_image, cmap="gray")
    plt.imshow(masked_roi, alpha=alpha, cmap=color)
    plt.axis('off')
    plt.title("mCherry "+sample+" cells", fontsize=fontsize)

    # Plotting all of the cells
    # Creating a mask so the image works better
    masked_roi_tot = np.ma.masked_where(A_reshape_tot_neuron == 0, A_reshape_tot_neuron)
    plt.subplot(1,3,3)
    plt.imshow(subtracted_image, cmap="gray")
    plt.imshow(masked_roi_tot, alpha=alpha, cmap=color)
    plt.axis('off')
    plt.title("All ROIs", fontsize=fontsize)     

    if show_figure:
        plt.show()
    plt.savefig(os.path.join(os.path.join(save_path, 'mcherry '+sample+' rois'), name+' mcherry '+sample+' rois'+'.png'),dpi=400, bbox_inches='tight')
    

def check_hull(p, hull):
    #used in select_mcherry_negative
    #taken from stack overflow, needs at least 4 points to create a hull
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,sp.Delaunay):
        hull = sp.Delaunay(hull)

    return hull.find_simplex(p)>=0

def create_mcherry_neg(rois, mcherry_neg_idx):
    #used in select_mcherry_negative
    #creates an image of the mcherry negative cells
    
    dims = np.shape(rois)[0]
    mcherry_neg = np.zeros((dims, dims))
    for i, idx in enumerate(mcherry_neg_idx):
        mcherry_neg += rois[:,:,mcherry_neg_idx[i]]
    
    return mcherry_neg
        
        
def select_mcherry_negative(rois, cell_idx_list, mcherry_pos_idx, mcherry_idx_list):
    #use the above helper fuctions to find the indices of the mcherry negative cells
    
    # find which peaks are within convex hull of the mcherry peaks
    in_hull_list = check_hull(cell_idx_list, mcherry_idx_list)
    in_hull_idx = np.where(in_hull_list == True)[0]
    
    mcherry_neg_idx = in_hull_idx[np.invert(np.in1d(in_hull_idx, mcherry_pos_idx))]
    
    mcherry_neg = create_mcherry_neg(rois, mcherry_neg_idx)
    
    return mcherry_neg_idx, mcherry_neg


def visualize_positive_negative_mcherry_cells(mcherry_pos, mcherry_neg, subtracted_image, name, save_path, show_figure):
    """
    Plots the subtracted image, the subtracted image with an overlay of the Mcherry positive cells,
    and the subtracted image with an overlay of all the ROIs

    ::Inputs::
        mcherry_pos :numpy array: - Image of all the positive MCherry cells
        mcherry_neg :numpy array: - Image of all the negative MCherry cells
        subtracted_image :numpy array: - Image after subtracting A_green_adjusted_median from A_red_adjusted_median
        name : string used for naming the image
        save_path : string used to determine where to save the image

    """
    plt.close()

    alpha=.5
    fontsize = 9
    
    masked_roi_pos = np.ma.masked_where(mcherry_pos == 0, mcherry_pos)
    masked_roi_neg = np.ma.masked_where(mcherry_neg == 0, mcherry_neg)

    # Plotting the subtracted image
    plt.subplot(1,3,1)
    plt.imshow(subtracted_image, cmap="gray")
    plt.imshow(masked_roi_pos, alpha=alpha, cmap="hot")
    plt.axis('off')
    plt.title("mCherry pos cells", fontsize=fontsize)

    # Plotting the positive cells
    # Creating a mask so the image works better
    
    plt.subplot(1,3,2)
    plt.imshow(subtracted_image, cmap="gray")
    plt.imshow(masked_roi_pos, alpha=alpha, cmap="hot")
    plt.imshow(masked_roi_neg, alpha=alpha, cmap="winter")
    plt.axis('off')
    plt.title("mCherry pos and neg cells", fontsize=fontsize)

    # Plotting all of the cells
    # Creating a mask so the image works better
    #masked_roi_tot = np.ma.masked_where(A_reshape_tot_neuron == 0, A_reshape_tot_neuron)
    plt.subplot(1,3,3)
    plt.imshow(subtracted_image, cmap="gray")
    plt.imshow(masked_roi_neg, alpha=alpha, cmap="winter")
    plt.axis('off')
    plt.title("mCherry neg cells", fontsize=fontsize)

    if show_figure:
        plt.show()
    plt.savefig(os.path.join(os.path.join(save_path, 'mcherry positive negative rois'), name+' mcherry positive negative rois'+'.png'),dpi=400, bbox_inches='tight')


def save_mcherry_hdf5(hdf5, mcherry_pos_idx, mcherry_neg_idx):
    # save the mcherry identifications in the hdf5 file
    
    # delete if prior data
    keys = ['tdtm', 'mcherry_pos','mcherry_neg']
    for key in keys:
        if key in hdf5.keys():
            del hdf5[key]
    
    temp_pos = np.zeros(np.shape(hdf5['traces'][()])[0])
    temp_neg = np.zeros(np.shape(hdf5['traces'][()])[0])
    
    temp_pos[mcherry_pos_idx] = 1
    temp_neg[mcherry_neg_idx] = -1
    
    temp1 = hdf5.create_dataset('mcherry_pos' ,data=temp_pos)
    temp2 = hdf5.create_dataset('mcherry_neg' ,data=temp_neg)
    
