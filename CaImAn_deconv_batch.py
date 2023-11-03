#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch pipeline for running CaImAn deconvolution on the Shared Computing Cluster

@author: jnorm
"""

import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
from PIL import Image
import h5py


try:
    
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params


from utilities.utilities import cleanUpMotionCorrection, cleanUpFinal, parseExperimentXML
from procedures.correctMotion import correctMotion1Channel
from procedures.correctMotion import correctMotion
from procedures.videoGeneration import generateMovie

import CaImAn_deconv_batch_utils as jutils

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                        "[%(process)d] %(message)s",
                        level=logging.DEBUG)

#%%
def main():
    pass  # For compatibility between running under Spyder and the CLI

#%% Set up list of folders to pass the function
    
    three_up_dir = '/projectnb/ndlpop/jfnorman/data/behavior 3/raw imaging data/'
    savePath = '/projectnb/ndlpop/jfnorman/data/behavior 3/output/'     # must have 'data', 'videos', 'ROI image', 'correlation image' folders

    mice = [f for f in os.listdir(three_up_dir) if os.path.isdir(os.path.join(three_up_dir, f))]

    for mouse in mice:
        
        two_up_dir = os.path.join(three_up_dir, mouse)
    
        days = [f for f in os.listdir(two_up_dir) if os.path.isdir(os.path.join(two_up_dir, f))]
        print(days)
        
        for day in days:
            one_up_dir = os.path.join(two_up_dir,day) 
    
            print(one_up_dir)
            folders = [f for f in os.listdir(one_up_dir) if os.path.isdir(os.path.join(one_up_dir, f))]
            print(folders)
            
            for folder in folders:
                loadPath = os.path.join(one_up_dir,folder)
                print(loadPath)
                
                saveName = mouse+" "+day+" "+folder[5:] #remove the .raw extension
                
            #%% Select file to be processed
                
                #if True, then figures open. If False, then no figures open
                num_colors = 1
                is_siri = False     #only used for siri project, where 2um puncta were investigated
                
                fPath = '/projectnb2/ndlpop/jfnorman/caimanfiles/JadScripts/temp/'    # working path to store temporary files and results
                xmlfile = os.path.join(loadPath, "Experiment.xml")      # .xml file that is produced with the .raw file
                fsave_gcamp = fPath + "gcamp.npy"
                fsave_tdtm = fPath + "tdTomato.npy"
                
                # filename to be processed
                fileList = [f for f in os.listdir(loadPath) if os.path.isfile(os.path.join(loadPath, f))]
                for f in fileList:
                    if f[-4:]=='.raw' and f[:5]=='Image':
                        fnames = os.path.join(loadPath,f)
                        logging.info('Data file location: ' +fnames)
                        del fileList
                        break
            #%%
                expInfo = jutils.parseExperimentXML(xmlfile)
            
                T = expInfo['nFrames']
                dims = (expInfo['width'], expInfo['height'])
            
                if os.path.isfile(fsave_gcamp) and os.path.isfile(fsave_tdtm):
                    processRawFile = False
                else:
                    processRawFile = True
            
                if processRawFile:
                    logging.info('Loading data from: ' +fnames)
                    A = np.fromfile(fnames, dtype='uint16', count=-1, sep="")
                    A = A.astype(np.float32)
                    if num_colors == 2:
                        A = A.reshape([2*T,dims[0],dims[1]],order='C') #use this if dual color
                        A_green = A[0::2,:,:]
                        A_red = A[1::2,:,:]
                        np.save(fsave_gcamp, A_green)
                        np.save(fsave_tdtm, A_red)
                        del A, A_green, A_red
                    if num_colors == 1:
                        A = A.reshape([T,dims[0],dims[1]],order='C') #use this if single color
                        np.save(fsave_gcamp, A)
                        A_red = np.zeros([T,dims[0],dims[1]])
                        np.save(fsave_tdtm, A_red) 
                        del A, A_red
                else:
                    logging.info('Data already loaded to temporary location')
            
                with open(fPath+'exp.txt', 'w') as file:
                    json.dump(expInfo, file)
            
                fnames = fsave_gcamp
                fnames_tdtm = fsave_tdtm
            
                #%% Setup some parameters for data and motion correction
            
                # dataset dependent parameters
                fr = expInfo['frameRate']             # imaging rate in frames per second
                decay_time = 0.3    # length of a typical transient in seconds
                dxy = (expInfo['pixelSizeUM'], expInfo['pixelSizeUM'])      # spatial resolution in x and y in (um per pixel)
            
                #look into these
                if expInfo['pixelSizeUM']<1:
                    max_shift_um = (6., 6.)       # maximum shift in um
                    patch_motion_um = (50., 50.) # patch size for non-rigid correction in um
                elif expInfo['pixelSizeUM']<2:
                    max_shift_um = (12., 12.)
                    patch_motion_um = (100., 100.)
                elif expInfo['pixelSizeUM']<3:
                    max_shift_um = (24., 24.)
                    patch_motion_um = (200., 200.)
                else:
                    max_shift_um = (12., 12.)
                    patch_motion_um = (100., 100.)
            
                # motion correction parameters
                pw_rigid = False       # flag to select rigid vs pw_rigid motion correction
                # maximum allowed rigid shift in pixels
                max_shifts = tuple([int(a/b) for a, b in zip(max_shift_um, dxy)])
                # start a new patch for pw-rigid motion correction every x pixels
                strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
                # overlap between pathes (size of patch in pixels: strides+overlaps)
                overlaps = (24, 24)
                # maximum deviation allowed for patch with respect to rigid shifts
                max_deviation_rigid = 3
                #set field of view size (JFN change)
                #dims = (300, 300)
            
                mc_dict = {
                    'fnames': fnames,
                    'fr': fr,
                    'decay_time': decay_time,
                    'dxy': dxy,
                    'pw_rigid': pw_rigid,
                    'max_shifts': max_shifts,
                    'strides': strides,
                    'overlaps': overlaps,
                    'max_deviation_rigid': max_deviation_rigid,
                    'border_nan': 'copy'
                }
            
                opts = params.CNMFParams(params_dict=mc_dict)
            
            # %% start a cluster for parallel processing
                c, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=None, single_thread=False)
                
            #%% MOTION CORRECTION
                # first we create a motion correction object with the specified parameters
                mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
                # note that the file is not loaded in memory
                #  Run (piecewise-rigid motion) correction using NoRMCorre
                mc.motion_correct(save_movie=True)
            
                mc_tdtm = MotionCorrect(fnames_tdtm, dview=dview, **opts.get_group('motion'))
                mc_tdtm.motion_correct(save_movie=True)
             
                # compare with original movie
                m_els = cm.load(mc.mmap_file)
            
                m_els_tdtm = cm.load(mc_tdtm.mmap_file)
            
            # %% produce the summary/stat images for the 2 channels
                tdtm_mean = m_els_tdtm.mean(axis=(0))
                tdtm_std = m_els_tdtm.std(axis=(0))  
                tdtm_max = m_els_tdtm.max(axis=0)
                tdtm_min = m_els_tdtm.min(axis=0)
                
                gcamp_mean = m_els.mean(axis=0)
                gcamp_std = m_els.std(axis=0)
                gcamp_max = m_els.max(axis=0)
                gcamp_min = m_els.min(axis=0)
                
            #%% MEMORY MAPPING
                border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
                # you can include the boundaries of the FOV if you used the 'copy' option
                # during motion correction, although be careful about the components near
                # the boundaries
            
                # memory map the file in order 'C'
                fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                                           border_to_0=border_to_0)  # exclude borders
            
                # now load the file
                Yr, dims, T = cm.load_memmap(fname_new)
                images = np.reshape(Yr.T, [T] + list(dims), order='F')
                # load frames in python format (T x X x Y)
            
            #%% restart cluster to clean up memory
                cm.stop_server(dview=dview)
                c, dview, n_processes = cm.cluster.setup_cluster(
                    backend='local', n_processes=None, single_thread=False)
            
            
            # %%  parameters for source extraction and deconvolution
                p = 2                    # order of the autoregressive system
                gnb = 2                 # number of global background components
                merge_thr = 0.85         # merging threshold, max correlation allowed
                rf = 25   # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
                stride_cnmf = 10          # amount of overlap between the patches in pixels
                #change this to 10/12
                K = 5                    # number of components per patch
                if expInfo['widthUM']<1000:
                    neuronUM = 20       # typical neuron size in um
                    if is_siri:
                        neuronUM = 3    # for SiRI experiments, use 3UM for puncta identification
                    neuronPx = np.floor(neuronUM*(expInfo['width']/expInfo['widthUM'])/2).astype(int)   # half neuron size in pz
                    gSig = [neuronPx, neuronPx]
                else:
                    gSig = [2, 2]            # expected half size of neurons in pixels
                # initialization method (if analyzing dendritic data using 'sparse_nmf')
                method_init = 'greedy_roi'
                ssub = 2                     # spatial subsampling during initialization
                tsub = 2                     # temporal subsampling during intialization
            
                # parameters for component evaluation
                opts_dict = {'fnames': fnames,
                             'fr': fr,
                             'nb': gnb,
                             'rf': rf,
                             'K': K,
                             'gSig': gSig,
                             'stride': stride_cnmf,
                             'method_init': method_init,
                             'rolling_sum': True,
                             'merge_thr': merge_thr,
                             'n_processes': n_processes,
                             'only_init': True,
                             'ssub': ssub,
                             'tsub': tsub}
            
                opts.change_params(params_dict=opts_dict)
            #%%  RUN CNMF ON PATCHES
                # First extract spatial and temporal components on patches and combine them
                # for this step deconvolution is turned off (p=0)
            
                opts.change_params({'p': 0})
                cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
                cnm = cnm.fit(images)
            
            #%%  plot contours of found components --- optional
                Cn = cm.local_correlations(images, swap_dim=False)
                Cn[np.isnan(Cn)] = 0
            
            #%%  RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
                cnm.params.change_params({'p': p})
                cnm2 = cnm.refit(images, dview=dview)
            #%%  COMPONENT EVALUATION
                # the components are evaluated in three ways:
                #   a) the shape of each component must be correlated with the data
                #   b) a minimum peak SNR is required over the length of a transient
                #   c) each shape passes a CNN based classifier
                min_SNR = 5  # signal to noise ratio for accepting a component
                rval_thr = 0.7  # space correlation threshold for accepting a component
                cnn_thr = 0.7  # threshold for CNN based classifier
                cnn_lowest = 0.15 # neurons with cnn probability lower than this value are rejected
            
                cnm2.params.set('quality', {'decay_time': decay_time,
                                           'min_SNR': min_SNR,
                                           'rval_thr': rval_thr,
                                           'use_cnn': True,
                                           'min_cnn_thr': cnn_thr,
                                           'cnn_lowest': cnn_lowest})
                cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
            
                #%% FINAL TRACES
            
                #update object with selected components
                cnm2.estimates.select_components(use_object=True)
                #Extract DF/F values
                cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)

            
                #%% STOP CLUSTER and clean up log files
                cm.stop_server(dview=dview)
                log_files = glob.glob('*_LOG_*')
                for log_file in log_files:
                    os.remove(log_file)
            
            #%%Create output
                
                if cnm2.estimates.idx_components != None:
                    #print('Using good components in idx')
                    indComponents = cnm2.estimates.idx_components
                else:
                    #print('Using all components')
                    indComponents = np.arange(cnm2.estimates.nr)
                
                numcells = len(indComponents)
                
                #store traces in separate array
                traces = np.append(np.array(range(1,T+1)), np.array(cnm2.estimates.F_dff))
                traces = np.reshape(traces,(numcells+1,T))
            
                #save correlation image (Cn) and static tdt image (tdtm_mean) as images
                Cn_image = (Cn*256).astype('uint8') #make 8 bit for cell counting
                Cn_image = Image.fromarray(Cn_image)    
                Cn_image.save(os.path.join(savePath,'correlation image/'+saveName+" correlation image.tif"))
            
                
                
                #%% save hdf5 file of estimates
                
                f1 = h5py.File(os.path.join(os.path.normpath(savePath),"data/"+saveName+".hdf5"), "w")
                A = f1.create_dataset("A", data=cnm2.estimates.A.toarray()) #these are the rolled ROIs
                b = f1.create_dataset("b", data=cnm2.estimates.b) #this is the background
                traces = f1.create_dataset("traces", data=cnm2.estimates.F_dff)
                C = f1.create_dataset("C", data=cnm2.estimates.C) #raw estimates for danny
                S = f1.create_dataset("S", data=cnm2.estimates.S) #these are the spike probabilities
                YrA = f1.create_dataset("YrA", data=cnm2.estimates.YrA) #these are the residuals
                tdtm = f1.create_dataset("tdtm", data=np.zeros(numcells)) #create this to be filled in later
                time = f1.create_dataset("time", data=(1/fr)*np.arange(T)) #time variable
                f1.close()
                

            #%% save traces and ROI images
                
                traces1 = cnm2.estimates.F_dff
                
                numcells1 = np.shape(traces1)[0]
                #T1 = np.shape(traces1)[1]
                excitatory_color = 'green'
                inhibitory_color = 'red'
                
                #separated is with the traces separated apart for visualization
                traces1_separated = traces1+np.reshape(np.repeat(np.arange(numcells1),T), (numcells1, T))
                
                #plot raw traces
                trace_fig1 = plt.figure()
                trace_ax1 = plt.axes()
                
                x = np.arange(numcells1)
                x = x[1:]
                labels = np.arange(T)
                for i in x:
                    clr = excitatory_color
                    trace_ax1.plot(labels/fr,traces1_separated[i,:], color=clr, linewidth=.6) 
                
                plt.ylim((0,numcells1))
                plt.xlim(0,T/fr)

                start, end = trace_ax1.get_xlim()
                #trace_ax1.xaxis.set_ticks(np.arange(start, end, 30))
                plt.xticks(fontsize=8)
                plt.xlabel('Time (seconds)',fontsize=14)
                plt.ylabel('Cell #', fontsize=14)
                plt.title(saveName,fontsize=20)

                trace_fig1.savefig(savePath+"traces/"+saveName+' traces.jpg',dpi=400)
                
                
                roi = cnm2.estimates.A.toarray().T
                num_pixels = np.sqrt(roi.shape[1]).astype(int)
                roi_sum = np.sum(roi, axis=0)
                roi_sum = np.reshape(roi_sum,(num_pixels, num_pixels))*num_pixels*2
                img = Image.fromarray(roi_sum)
                img = img.convert("L")
                
                img.save(savePath+"rois/"+saveName+" ROI image.jpg")
                
            #%% make and save videos
                
                fPath2 = '/projectnb2/ndlpop/jfnorman/caimanfiles/JadScripts/temp2/'    # working path to store temporary files and results
                speed = 5
                state ="_"
                fname_memmap = correctMotion1Channel(loadPath, fPath2, folder, state,
                                                      display_images_mc=False,
                                                      display_summary_images=False)
    
                vidName = os.path.join(fPath2, saveName+" movie_x"+str(speed)+".mp4")
                
                videoFile = generateMovie(fname_memmap, fPath2, videoFile=vidName, fr=fr, dims=dims,
                         qmin=0.5, qmax=99.5, magnification=1, ds_ratio=1/speed, gain=1.)
                
                
                cleanUpFinal(fPath2, savePath+"videos/", "M", inplace = False)  
    
            #%% DELETE TEMPORARY AND MEMORY MAPPED FILES
            
                del(images)
                del(dims)
                del(T)
                del(Yr)
                
            #%% close plots and delete files
                
                #plt.close('all')
                jutils.delTempFiles(fPath)

# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()
#