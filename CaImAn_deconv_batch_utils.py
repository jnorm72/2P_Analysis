#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to help parse the metadata and to clean up the deconvolution

@author: jad, jnorm
"""

import xml.etree.ElementTree as ET
import inspect
import cv2
import h5py
import logging
import numpy as np
import os
import pylab as pl
import scipy
from scipy.sparse import spdiags, issparse, csc_matrix, csr_matrix
import scipy.ndimage.morphology as morph
from skimage.feature.peak import _get_high_intensity_peaks
import tifffile
from typing import List


# functions

def parseExperimentXML(xmlfile): 
    """ Return the experimental expInformation
    Args:
        xmlfile: full address of Experiment.xml file

    Returns:
        expInfo: dictionary containing experimental information

    See Also:
    """

    tree = ET.parse(xmlfile) 
    root = tree.getroot() 
    expInfo = {}
    
    # get file name
    for child in root.findall('Name'):
        expInfo['name'] = child.attrib['name']
    # get experiment date/time
    for child in root.findall('Date'):
        expInfo['date'] = child.attrib['date']
    # get magnification
    for child in root.findall('Magnification'):
        expInfo['mag'] = child.attrib['mag']
    # imaging depth and stage location
    for child in root.findall('ZStage'):
        expInfo['depthUM'] = np.abs(np.float(child.attrib['setupPositionMM'])*1000)
    for child in root.findall('Sample'):
        expInfo['StageX_UM']=np.float(child.attrib['initialStageLocationX'])*1000;
        expInfo['StageY_UM']=np.float(child.attrib['initialStageLocationY'])*1000;
    # Imaging expInformation
    for child in root.findall('LSM'):
        expInfo['width'] = np.int(child.attrib['pixelX'])
        expInfo['height'] = np.int(child.attrib['pixelY'])
        expInfo['pixelSizeUM'] = np.double(child.attrib['pixelSizeUM'])
        expInfo['widthUM'] = np.double(child.attrib['widthUM'])
        expInfo['heightUM'] = np.double(child.attrib['heightUM'])
        if np.int(child.attrib['averageMode'])==0:
            expInfo['frameRate'] = np.double(child.attrib['frameRate'])
        elif np.int(child.attrib['averageMode'])==1:
            expInfo['frameRate'] = np.double(child.attrib['frameRate'])/np.double(child.attrib['averageNum'])
    # number of frames
    for child in root.findall('Streaming'):
        expInfo['nFrames'] = np.int(child.attrib['frames'])
     # get notes
    for child in root.findall('ExperimentNotes'):
        expInfo['notes'] = child.attrib['text'] 
    
    return expInfo 




def delTempFiles(path):
    """ Removes all temporary nmpy and memmap files
    Args:
        path: path to the directory that needs cleaning

    Returns:
        none

    See Also:
    """
    
    fileList = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    for f in fileList:
        if f[-5:]=='.mmap':
            os.remove(os.path.join(path, f))        
        elif f == 'gcamp.npy':
            os.remove(os.path.join(path, f))
        elif f == 'tdTomato.npy':
            os.remove(os.path.join(path, f))


    