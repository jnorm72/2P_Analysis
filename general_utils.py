# -*- coding: utf-8 -*-
"""
Generally applicable functions such as finding and loading files

@author: jnorm
"""
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
import cv2


# functions

def find_hdf5(file_path):
    #takes in the file path and returns an array of the hdf5s
    file_array = []
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename) 
        if ".hdf5" in f:
            file_array.append(f)
    return file_array

def find_abf(file_path):
    #takes in the file path and returns an array of the abfs
    file_array = []
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename) 
        if ".abf" in f:
            file_array.append(f)
    return file_array

def find_file(file_path):
    #takes in the file path and returns an array of the files
    file_array = []
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename)
        file_array.append(f)
    return file_array


def load_npy(file_path, width):
    #will load all numpy files in the given folder by stacking them vertically
    
    temp = [f for f in os.listdir(file_path) if f[-4:]=='.npy']
    full_data = np.zeros(width)
    for f in temp:
        print(f)
        full_data = np.vstack([full_data, np.load(os.path.join(file_path,f))])
    #remove padding
    full_data = full_data[1:,:]
    
    return full_data

def load_raw(file_path):
    # assumes dual color imaging, 100 frames, and 512x512 imaging. Easy to update these into variables
    fileList = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    for f in fileList:
        if f[-4:]=='.raw' and f[:5]=='Image':
            fnames = os.path.join(file_path,f)
    
    #hard code in the dimensions of red file, can make this a variable if needed in the future
    T = 100
    dims = (512, 512)
    
    A = np.fromfile(fnames, dtype='uint16', count=-1, sep="")
    A = A.astype(np.float32)
    
    A = A.reshape([2*T,dims[0],dims[1]],order='C') #use this if dual color
    A_green = A[0::2,:,:]
    A_red = A[1::2,:,:]
    
    return A_green, A_red


def load_tif(file_path):
    # takes in a tif file, and returns a numpy array
    
    cn_matrix = cv2.imread(file_path)
    
    #pull just the red channel - for grayscale, all the same
    cn_matrix = cn_matrix[:,:,0]
    
    return cn_matrix


#helper function to determine if something is a dataset
def is_dataset(hdf5):
    is_dataset = isinstance(hdf5, h5py.Dataset)
    return is_dataset 
