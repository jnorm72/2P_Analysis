# -*- coding: utf-8 -*-
"""
Functions to aggregate the data and to plot the data (with statistics)

@author: jnorm
"""
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import h5py
from decimal import Decimal


# functions

def agg_pickle(file_array):
    # takes in an array of pickled data and returns a numpy array
    
    firingrate_data = pd.read_pickle(file_array[0])
    for i, file in enumerate(file_array[1:]):
        temp = pd.read_pickle(file_array[i+1])
        firingrate_data = pd.merge(firingrate_data, temp, how='outer')
        print(file)
        print(i)
    return firingrate_data



def extract_3_epochs(time, stims_binary):
    #finds the baseline, stim, and no stim indices for a given file
    
    stim_idx = np.where(stims_binary==1)
    nostim_idx = np.where(stims_binary==0)
    early_idx = np.where(time<60)
    notlate_idx = np.where(time<300)
    notearly_idx = np.where(time>60)
    
    baseline_idx = np.intersect1d([nostim_idx],[early_idx])
    mid_idx = np.intersect1d([notlate_idx],[notearly_idx])
    nostim_mid_idx = np.intersect1d([mid_idx],[nostim_idx])
    stim_idx = stim_idx[0]
    
    return baseline_idx, stim_idx, nostim_mid_idx

def extract_stim_epochs(time, stims_binary,m223=False):
    #finds the baseline, stim, and no stim indices for a given file
    
    stim = np.where(stims_binary==1)
    nostim = np.where(stims_binary==0)
    nostim_time = np.where(time<240)
    poststim_time = np.where(time>240)
    second_stim_time = np.where(time>420)
    if m223:
        second_stim_time = np.where(time<420)
    third_nostim_time = np.where(time>600)
    
    nostim_idx = np.intersect1d([nostim],[nostim_time])
    poststim_idx = np.intersect1d([nostim],[poststim_time])
    second_stim_idx = np.intersect1d([stim],[second_stim_time])
    third_nostim_idx = np.intersect1d([nostim],[third_nostim_time])
    
    stim_idx = stim[0]
    
    
    return nostim_idx, stim_idx, poststim_idx, second_stim_idx, third_nostim_idx

def extract_lick_epochs(time, licks):
    #finds the baseline, stim, and no stim indices for a given file

    nolicks_idx = np.where(licks==0)    
    licks_idx = np.where(licks==1)

    nolicks_idx = nolicks_idx[0]
    licks_idx = licks_idx[0]
    
    return nolicks_idx, licks_idx

    
    
def boxplot_2(data1, data2, labels, title, ylabel, save_name, save_path, save_figure, is_violin=False, high=2, low=0, color1=[0,0,1,.5], color2=[0,1,0,.5], paired=False):
    
    font = 'Arial'
    dot_transparency = .5
    dot_color = 'black'
    dot_size = 40
    box1_color = color1
    box2_color = color2
    
    #plotting
    data1_box_pos = 1
    data2_box_pos = 2
    
    x1 = np.random.normal(data1_box_pos, 0.01, size=len(data1))
    x2 = np.random.normal(data2_box_pos, 0.01, size=len(data2))

    #creat plots
    fig1, ax1 = plt.subplots(figsize=(5,6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x', length=0)
    
    plt.scatter(x1, data1,alpha=dot_transparency,color=dot_color,s=dot_size)
    plt.scatter(x2, data2,alpha=dot_transparency,color=dot_color,s=dot_size)
    
    if is_violin:
        bp = plt.violinplot([data1,data2],positions=[data1_box_pos,data2_box_pos], showextrema=False, showmeans=True)
        #bp = plt.violinplot([data1,data2],positions=[data1_box_pos,data2_box_pos], showextrema=False)
        string = 'bodies'
    else:
        bp = plt.boxplot([data1, data2],positions=[data1_box_pos,data2_box_pos],notch=False,showfliers=False,patch_artist=True,boxprops=dict(linewidth=2),whiskerprops=dict(linewidth=2),medianprops=dict(linewidth=2,color='black'),capprops=dict(linewidth=2))
        string = 'boxes'
        
    colors = [box1_color, box2_color]
    
    for patch, color in zip(bp[string], colors):
        patch.set_facecolor(color)
    
    plt.ylabel(ylabel,fontname=font,fontsize=15)
    plt.xticks((data1_box_pos,data2_box_pos),labels,fontname=font,fontsize=18)
    plt.yticks(fontsize=12,fontname=font)
    plt.ylim(ymin=low,ymax=high)
    plt.title(title,fontname=font,fontsize=20)
    

    if paired:
        (test,p) = stats.wilcoxon(data1,data2)
        print('wilcoxon signed rank test')
    else:
        (test,p) = stats.mannwhitneyu(data1,data2)
        print('mannwhitney u test')

    print(np.mean(data1))
    print(np.mean(data2))
    print(p)
    
    bar_pos = high*.9
    high = high*.93
    if p<.001:
        ax1.text((data1_box_pos+data2_box_pos)/2,high,'***p='+str('%.2E' % Decimal(p)),fontname=font,fontsize=16,horizontalalignment='center')
    elif p<.05:
        ax1.text((data1_box_pos+data2_box_pos)/2,high,'*p='+str(np.around(p,3)),fontname=font,fontsize=16,horizontalalignment='center')
    else:
        ax1.text((data1_box_pos+data2_box_pos)/2,high,'p='+str(np.around(p,3)),fontname=font,fontsize=16,horizontalalignment='center')
    plt.plot([data1_box_pos,data2_box_pos,data1_box_pos,data2_box_pos],[bar_pos,bar_pos,bar_pos,bar_pos],linewidth=3,color='black')
    
    #save figure?
    if save_figure:
        plt.savefig(os.path.join(save_path, (save_name+'.png')),dpi=400, bbox_inches='tight')
    plt.show()
    
    
def boxplot_2_days(data1, data2, labels, title, ylabel, save_name, save_path, save_figure, is_violin=False, high=2, low=0, color1=[0,0,1,.5], color2=[0,1,0,.5], paired=False):
    
    font = 'Arial'
    dot_transparency = .5
    dot_color = 'black'
    dot_size = 40
    box1_color = color1
    box2_color = color2
    #ylabels = ['$R_{Stim}<<R_{No Stim}$', '$R_{Stim}=R_{No Stim}$', '$R_{Stim}>>R_{No Stim}$']
    ylabels = ['$R_{LC}<<R_{No LC}$', '$R_{LC}=R_{No LC}$', '$R_{LC}>>R_{No LC}$']
    
    #plotting
    data1_box_pos = 1
    data2_box_pos = 2
    
    x1 = np.random.normal(data1_box_pos, 0.01, size=len(data1))
    x2 = np.random.normal(data2_box_pos, 0.01, size=len(data2))

    #creat plots
    fig1, ax1 = plt.subplots(figsize=(5,6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x', length=0)
    
    plt.scatter(x1, data1,alpha=dot_transparency,color=dot_color,s=dot_size)
    plt.scatter(x2, data2,alpha=dot_transparency,color=dot_color,s=dot_size)
    
    if is_violin:
        bp = plt.violinplot([data1,data2],positions=[data1_box_pos,data2_box_pos], showextrema=False, showmeans=True)
        #bp = plt.violinplot([data1,data2],positions=[data1_box_pos,data2_box_pos], showextrema=False)
        string = 'bodies'
    else:
        bp = plt.boxplot([data1, data2],positions=[data1_box_pos,data2_box_pos],notch=False,showfliers=False,patch_artist=True,boxprops=dict(linewidth=2),whiskerprops=dict(linewidth=2),medianprops=dict(linewidth=2,color='black'),capprops=dict(linewidth=2))
        string = 'boxes'
        
    colors = [box1_color, box2_color]
    
    for patch, color in zip(bp[string], colors):
        patch.set_facecolor(color)
    
    #plt.ylabel(ylabel,fontname=font,fontsize=15)
    plt.xticks((data1_box_pos,data2_box_pos),labels,fontname=font,fontsize=18)
    plt.yticks([0,.5, 1], ylabels,fontsize=12,fontname=font)
    plt.ylim(ymin=low,ymax=high)
    plt.title(title,fontname=font,fontsize=20)
    plt.axhline(y=0.5, color='y', linestyle='--')
    
    

    if paired:
        (test,p) = stats.wilcoxon(data1,data2)
        print('wilcoxon signed rank test')
    else:
        (test,p) = stats.mannwhitneyu(data1,data2)
        print('mannwhitney u test')

    print(np.mean(data1))
    print(np.mean(data2))
    print(p)
    
    bar_pos = high*.9
    high = high*.93
    if p<.001:
        ax1.text((data1_box_pos+data2_box_pos)/2,high,'***p='+str('%.2E' % Decimal(p)),fontname=font,fontsize=16,horizontalalignment='center')
    elif p<.05:
        ax1.text((data1_box_pos+data2_box_pos)/2,high,'*p='+str(np.around(p,3)),fontname=font,fontsize=16,horizontalalignment='center')
    else:
        ax1.text((data1_box_pos+data2_box_pos)/2,high,'p='+str(np.around(p,3)),fontname=font,fontsize=16,horizontalalignment='center')
    plt.plot([data1_box_pos,data2_box_pos,data1_box_pos,data2_box_pos],[bar_pos,bar_pos,bar_pos,bar_pos],linewidth=3,color='black')
    
    plt.tight_layout()
    #save figure?
    if save_figure:
        plt.savefig(os.path.join(save_path, (save_name+'.png')),dpi=400)
    plt.show()


    
    
def boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=True, high=2, low=0, color1=[0,0,1,.5], color2=[0,1,0,.5], color3=[1,0,0,.5], paired=False, log=False):
    
    font = 'Arial'
    dot_transparency = .5
    dot_color = 'black'
    dot_size = 40
    box1_color = color1
    box2_color = color2
    box3_color = color3
    bar_pos = high
    
    
    #plotting
    data1_box_pos = 1
    data2_box_pos = 2
    data3_box_pos = 3
    
    x1 = np.random.normal(data1_box_pos, 0.01, size=len(data1))
    x2 = np.random.normal(data2_box_pos, 0.01, size=len(data2))
    x3 = np.random.normal(data3_box_pos, 0.01, size=len(data3))
    
    #creat plots
    fig1, ax1 = plt.subplots(figsize=(5,6))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(axis='x', length=0)
    
    plt.scatter(x1, data1,alpha=dot_transparency,color=dot_color,s=dot_size)
    plt.scatter(x2, data2,alpha=dot_transparency,color=dot_color,s=dot_size)
    plt.scatter(x3, data3,alpha=dot_transparency,color=dot_color,s=dot_size)
    
    if is_violin:
        bp = plt.violinplot([data1, data2, data3],positions=[data1_box_pos,data2_box_pos,data3_box_pos], showmeans='True')
        string = 'bodies'
    else:
        bp = plt.boxplot([data1, data2, data3],positions=[data1_box_pos,data2_box_pos,data3_box_pos],notch=False,showfliers=False,patch_artist=True,boxprops=dict(linewidth=2),whiskerprops=dict(linewidth=2),medianprops=dict(linewidth=2,color='black'),capprops=dict(linewidth=2))
        string = 'boxes'
    
    colors = [box1_color, box2_color, box3_color]
    for patch, color in zip(bp[string], colors):
        patch.set_facecolor(color)
    
    plt.ylabel(ylabel,fontname=font,fontsize=15)
    plt.xticks((data1_box_pos,data2_box_pos,data3_box_pos),labels,fontname=font,fontsize=18)
    plt.yticks(fontsize=12,fontname=font)
    if log:
        plt.yscale('log')
    plt.ylim(ymin=low,ymax=high)
    plt.title(title,fontname=font,fontsize=20)
    
    if paired:
        print('wilcoxon signed rank test')
        (test1,p1) = stats.wilcoxon(data1,data2)
        (test2,p2) = stats.wilcoxon(data1,data3)
        (test3,p3) = stats.wilcoxon(data2,data3)  
    else:
        print('mannwhitney u test')
        (test1,p1) = stats.mannwhitneyu(data1,data2)
        (test2,p2) = stats.mannwhitneyu(data1,data3)
        (test3,p3) = stats.mannwhitneyu(data2,data3)
        
        # (test1,p1) = stats.ttest_ind(data1,data2)
        # (test2,p2) = stats.ttest_ind(data1,data3)
        # (test3,p3) = stats.ttest_ind(data2,data3)
        
    p1 = p1*3
    p2 = p2*3
    p3 = p3*3
        
    print(np.mean(data1))
    print(np.mean(data2))
    print(np.mean(data3))
    print(p1)
    print(p2)
    print(p3)
    
    bar_pos1 = high*.9
    high1 = high*.8
    
    #p2
    if p2<.001:
        ax1.text((data1_box_pos+data3_box_pos)/2,high1,'***p='+str('%.2E' % Decimal(p2)),fontname=font,fontsize=16,horizontalalignment='center')
    elif p2<.05:
        ax1.text((data1_box_pos+data3_box_pos)/2,high1,'*p='+str(np.around(p2,3)),fontname=font,fontsize=16,horizontalalignment='center')
    else:
        ax1.text((data1_box_pos+data3_box_pos)/2,high1,'p='+str(np.around(p2,3)),fontname=font,fontsize=16,horizontalalignment='center')
    plt.plot([data1_box_pos,data3_box_pos,data1_box_pos,data3_box_pos],[bar_pos1,bar_pos1,bar_pos1,bar_pos1],linewidth=3,color='black')
    
    #p1
    bar_pos2 = bar_pos*.75
    high2 = high*.65
    if p1<.001:
        ax1.text((data1_box_pos+data2_box_pos-.3)/2,high2,'***p='+str('%.2E' % Decimal(p1)),fontname=font,fontsize=16,horizontalalignment='center')
    elif p1<.05:
        ax1.text((data1_box_pos+data2_box_pos)/2,high2,'*p='+str(np.around(p1,3)),fontname=font,fontsize=16,horizontalalignment='center')
    else:
        ax1.text((data1_box_pos+data2_box_pos)/2,high2,'p='+str(np.around(p1,3)),fontname=font,fontsize=16,horizontalalignment='center')
    plt.plot([data1_box_pos,data2_box_pos,data1_box_pos,data2_box_pos],[bar_pos2,bar_pos2,bar_pos2,bar_pos2],linewidth=3,color='black')
    
    #p3
    bar_pos3 = bar_pos*.65
    high3 = high*.55
    if p3<.001:
        ax1.text((data2_box_pos+data3_box_pos)/2,high3,'***p='+str('%.2E' % Decimal(p3)),fontname=font,fontsize=16,horizontalalignment='center')
    elif p3<.05:
        ax1.text((data2_box_pos+data3_box_pos)/2,high3,'*p='+str(np.around(p3,3)),fontname=font,fontsize=16,horizontalalignment='center')
    else:
        ax1.text((data2_box_pos+data3_box_pos)/2,high3,'p='+str(np.around(p3,3)),fontname=font,fontsize=16,horizontalalignment='center')
    plt.plot([data2_box_pos,data3_box_pos,data2_box_pos,data3_box_pos],[bar_pos3,bar_pos3,bar_pos3,bar_pos3],linewidth=3,color='black')

    
    #save figure?
    if save_figure:
        plt.savefig(os.path.join(save_path,save_name+'.png'),dpi=400, bbox_inches='tight')
    plt.show()    
    

    
# bonus functions if pie plots are desired

def pieplot_2(data, labels, title, save_path, save_figure):
    
    #plot a pie plot with the first data point exploded
    
    font = 'Arial'
    
    explode = np.zeros(len(labels))
    explode[0] = .1

    fig1, ax1 = plt.subplots()
    ax1.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.title(title,fontname=font,fontsize=20)
    
    #save figure?
    if save_figure:
        plt.savefig(os.path.join(save_path,title+'.png'),dpi=400, bbox_inches='tight')
        
    plt.show()
        
def pieplot_3(data, labels, title, save_path, save_figure):
    
    #plot a pie plot with the first data point exploded
    
    font = 'Arial'
    
    explode = np.zeros(len(labels))
    explode[2] = .1

    fig1, ax1 = plt.subplots()
    ax1.pie(data, explode=explode, autopct='%1.1f%%',
            shadow=False, startangle=90, textprops={'size': 'x-large'})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.title(title,fontname=font,fontsize=20)
    
    #save figure?
    if save_figure:
        plt.savefig(os.path.join(save_path,'Control '+title+' stim modulation.png'),dpi=400, bbox_inches='tight')
        
    plt.show()  
    
    