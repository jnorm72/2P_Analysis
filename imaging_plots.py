# -*- coding: utf-8 -*-
"""
Removes lick modulated cells, and then plots and runs statistics on each desired comparison

@author: jnorm
"""


# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.stats as sp
import general_utils as gen
import agg_plot_utils as agg



#%% load data 

good_mice = ['M271','M273','M274','M275','M282','M286'] #control are M271, M275, M282
file_path = r'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/firing rate comparison/data/'

file_array = gen.find_file(file_path)

firingrate_data = agg.agg_pickle(file_array)
print(file_array)
firingrate_data = firingrate_data[firingrate_data['Mouse'].isin(good_mice)]


#%% determine which cells are modulated by licking and remove them

lick_mod = firingrate_data['lick rate']/(firingrate_data['no lick rate']+.001)

# set threshold
lick_mod_thresh = 2.5
print(lick_mod_thresh)

# binarize between threshold and 20
firingrate_data['lick mod'] = (lick_mod>lick_mod_thresh) & (lick_mod<20)
firingrate_data['lick mod neg'] = lick_mod<(1/lick_mod_thresh)

# remove cells that are modulated by licking
firingrate_data = firingrate_data.loc[firingrate_data['lick mod']==False]
firingrate_data = firingrate_data.loc[firingrate_data['lick mod neg']==False]

#%% set up variables for plotting

hab = firingrate_data.loc[firingrate_data['Day'] == '-1']
day1 = firingrate_data.loc[firingrate_data['Day'] == '1']

exp = firingrate_data.loc[(firingrate_data['Experimental'] == '1') & (firingrate_data['Day'] == '1')]
exp_hab = firingrate_data.loc[(firingrate_data['Experimental'] == '1') & (firingrate_data['Day'] == '-1')]
noexp = firingrate_data.loc[(firingrate_data['Experimental'] == '0') & (firingrate_data['Day'] == '1')]
noexp_hab = firingrate_data.loc[(firingrate_data['Experimental'] == '0') & (firingrate_data['Day'] == '-1')]

eng_pos = firingrate_data.loc[firingrate_data['mCherry'] == '1.0']
eng_neg = firingrate_data.loc[firingrate_data['mCherry'] == '-1.0']

eng_pos_exp = eng_pos.loc[(eng_pos['Experimental'] == '1') & (eng_pos['Day'] == '1')]
eng_neg_exp = eng_neg.loc[(eng_neg['Experimental'] == '1') & (eng_neg['Day'] == '1')]

eng_pos_noexp = eng_pos.loc[(eng_pos['Experimental'] == '0') & (eng_pos['Day'] == '1')]
eng_neg_noexp = eng_neg.loc[(eng_neg['Experimental'] == '0') & (eng_neg['Day'] == '1')]


day2 = firingrate_data.loc[firingrate_data['Day'] == '2']
day3 = firingrate_data.loc[firingrate_data['Day'] == '3']

exp2 = firingrate_data.loc[(firingrate_data['Experimental'] == '1') & (firingrate_data['Day'] == '2')]
exp3 = firingrate_data.loc[(firingrate_data['Experimental'] == '1') & (firingrate_data['Day'] == '3')]
noexp2 = firingrate_data.loc[(firingrate_data['Experimental'] == '0') & (firingrate_data['Day'] == '2')]
noexp3 = firingrate_data.loc[(firingrate_data['Experimental'] == '0') & (firingrate_data['Day'] == '3')]

eng_pos_exp2 = eng_pos.loc[(eng_pos['Experimental'] == '1') & (eng_pos['Day'] == '2')]
eng_neg_exp2 = eng_neg.loc[(eng_neg['Experimental'] == '1') & (eng_neg['Day'] == '2')]
eng_pos_exp3 = eng_pos.loc[(eng_pos['Experimental'] == '1') & (eng_pos['Day'] == '3')]
eng_neg_exp3 = eng_neg.loc[(eng_neg['Experimental'] == '1') & (eng_neg['Day'] == '3')]

eng_pos_noexp2 = eng_pos.loc[(eng_pos['Experimental'] == '0') & (eng_pos['Day'] == '2')]
eng_neg_noexp2 = eng_neg.loc[(eng_neg['Experimental'] == '0') & (eng_neg['Day'] == '2')]
eng_pos_noexp3 = eng_pos.loc[(eng_pos['Experimental'] == '0') & (eng_pos['Day'] == '3')]
eng_neg_noexp3 = eng_neg.loc[(eng_neg['Experimental'] == '0') & (eng_neg['Day'] == '3')]


#%% neutral memory stimulation

plt.close('all')

save_figure = True
save_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/manuscript figures/8-4-23/'
ylabel = 'Firing Rate (putative spikes/sec)'
labels = ('No Stim', 'Stim')
high = 5
low = 0
is_violin = True

title = 'Population'
color1 = [.5,.5,.5,.7]
color2 = [.7,.5,.5,.7]
save_name = 'Control stim vs no stim'
agg.boxplot_2(noexp['no stim rate'], noexp['stim rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)

title = 'MAN+'
color1 = [.5,.5,.5,.7]
color2 = [.7,.5,.5,.7]
save_name = 'Control Engram+ stim vs no stim'
agg.boxplot_2(eng_pos_noexp['no stim rate'], eng_pos_noexp['stim rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)

title = 'MAN-'
color1 = [.5,.5,.5,.7]
color2 = [.7,.5,.5,.7]
save_name = 'Control Engram- stim vs no stim'
agg.boxplot_2(eng_neg_noexp['no stim rate'], eng_neg_noexp['stim rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)



#%% fear memory stimulation

save_figure = True
save_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/manuscript figures/8-4-23/'
ylabel = 'Firing Rate (putative spikes/sec)'
labels = ('No Stim', 'Stim')
high = 5
low = 0
is_violin = True

title = 'Population'
color1 = [0,1,.5,.7]
color2 = [.6,1,.5,.8]
save_name = 'Experimental stim vs no stim'
agg.boxplot_2(exp['no stim rate'], exp['stim rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)

title = 'MAN+'
color1 = [0,1,.5,.7]
color2 = [.6,1,.5,.8]
save_name = 'Experimental Engram+ stim vs no stim'
agg.boxplot_2(eng_pos_exp['no stim rate'], eng_pos_exp['stim rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)

title = 'MAN-'
color1 = [0,1,.5,.7]
color2 = [.6,1,.5,.8]
save_name = 'Experimental Engram- stim vs no stim'
agg.boxplot_2(eng_neg_exp['no stim rate'], eng_neg_exp['stim rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)

plt.close('all')


#%% behavior activity

plt.close('all')

save_figure = True
save_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/manuscript figures/8-4-23/supplementary/'
ylabel = 'Firing Rate (putative spikes/sec)'
labels = ('Light Induced Freezing', 'No Light Induced Freezing')
high = 5
low = 0
is_violin = True


color1 = [.6,1,.7,.8]
color2 = [.6,1,.5,.8]

title = 'Population'
save_name = 'Population behavior activity'
agg.boxplot_2(exp['second stim no lick rate'], exp['second stim lick rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)

title = 'MAN+'
save_name = 'Engram+ behavior activity'
agg.boxplot_2(eng_pos_exp['second stim no lick rate'], eng_pos_exp['second stim lick rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)

title = 'MAN-'
save_name = 'Engram- behavior activity'
agg.boxplot_2(eng_neg_exp['second stim no lick rate'], eng_neg_exp['second stim lick rate'], labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, paired=True)


#plt.close('all')

#%% neutral memory division across days

plt.close('all')

save_figure = True
save_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/manuscript figures/8-4-23/'
ylabel = 'Change in Firing Rate'
labels = ('Day 1', 'Day 2', 'Day 3')
labels1 = ('Day 1', 'Subsequent Days')
high = 1
low = 0
is_violin = True


title = 'Population'
color1 = [.2,.2,.2,.7]
color2 = [.5,.5,.5,.35]
color3 = [.6,.6,.6,.3]
data1 = (noexp['stim rate'] / (noexp['no stim rate']+noexp['stim rate']+.000001)).array
data2 = (noexp2['stim rate'] / (noexp2['no stim rate']+noexp2['stim rate']+.000001)).array
data3 = (noexp3['stim rate'] / (noexp3['no stim rate']+noexp3['stim rate']+.000001)).array
save_name = 'Control Population Across Days stim vs no stim'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3, log=log)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color3)

title = 'MAN+'
data1 = (eng_pos_noexp['stim rate'] / (eng_pos_noexp['no stim rate']+eng_pos_noexp['stim rate']+.000001)).array
data2 = (eng_pos_noexp2['stim rate'] / (eng_pos_noexp2['no stim rate']+eng_pos_noexp2['stim rate']+.000001)).array
data3 = (eng_pos_noexp3['stim rate'] / (eng_pos_noexp3['no stim rate']+eng_pos_noexp3['stim rate']+.000001)).array
save_name = 'Control Engram+ Across Days stim vs no stim'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color3)


title = 'MAN-'
data1 = (eng_neg_noexp['stim rate'] / (eng_neg_noexp['no stim rate']+eng_neg_noexp['stim rate']+.000001)).array
data2 = (eng_neg_noexp2['stim rate'] / (eng_neg_noexp2['no stim rate']+eng_neg_noexp2['stim rate']+.000001)).array
data3 = (eng_neg_noexp3['stim rate'] / (eng_neg_noexp3['no stim rate']+eng_neg_noexp3['stim rate']+.000001)).array
save_name = 'Control Engram- Across Days stim vs no stim'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color3)


#plt.close('all')

#%% fear memory across days division
plt.close('all')

save_figure = True
save_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/manuscript figures/8-4-23/'
ylabel = 'Change in Firing Rate'
labels = ('Day 1', 'Day 2', 'Day 3')
labels1 = ('Day 1', 'Subsequent Days')
high = 1
low = 0
is_violin = True


title = 'Population'
color1 = [0,1,.5,.7]
color2 = [0,.6,.2,.3]
color3 = [0,.6,.2,.3]
data1 = (exp['stim rate'] / (exp['no stim rate']+exp['stim rate']+.000001)).array
data2 = (exp2['stim rate'] / (exp2['no stim rate']+exp2['stim rate']+.000001)).array
data3 = (exp3['stim rate'] / (exp3['no stim rate']+exp3['stim rate']+.000001)).array
save_name = 'Experimental Population Across Days stim vs no stim'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2)

title = 'MAN+'
data1 = (eng_pos_exp['stim rate'] / (eng_pos_exp['no stim rate']+eng_pos_exp['stim rate']+.000001)).array
data2 = (eng_pos_exp2['stim rate'] / (eng_pos_exp2['no stim rate']+eng_pos_exp2['stim rate']+.000001)).array
data3 = (eng_pos_exp3['stim rate'] / (eng_pos_exp3['no stim rate']+eng_pos_exp3['stim rate']+.000001)).array
save_name = 'Experimental Engram+ Across Days stim vs no stim'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2)

title = 'MAN-'
data1 = (eng_neg_exp['stim rate'] / (eng_neg_exp['no stim rate']+eng_neg_exp['stim rate']+.000001)).array
data2 = (eng_neg_exp2['stim rate'] / (eng_neg_exp2['no stim rate']+eng_neg_exp2['stim rate']+.000001)).array
data3 = (eng_neg_exp3['stim rate'] / (eng_neg_exp3['no stim rate']+eng_neg_exp3['stim rate']+.000001)).array
save_name = 'Experimental Engram- Across Days stim vs no stim'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2)

#plt.close('all')



#%% behavior activity division across days

plt.close('all')

save_figure = True
save_path = 'D:/White Lab Data/Simultaneous stimulation and imaging/stim and imaging behavior (red)/aggregated imaging data/figures/manuscript figures/8-4-23/'
ylabel = 'Change in Firing Rate'
labels = ('Day 1', 'Day 2', 'Day 3')
labels1 = ('Day 1', 'Subsequent Days')
high = 1
low = 0
is_violin = True



title = 'Population'
color1 = [0,1,.5,.7]
color2 = [0,.6,.2,.3]
color3 = [0,.6,.2,.3]
data1 = (exp['second stim lick rate'] / (exp['second stim no lick rate']+exp['second stim lick rate']+.000001)).array
data2 = (exp2['second stim lick rate'] / (exp2['second stim no lick rate']+exp2['second stim lick rate']+.000001)).array
data3 = (exp3['second stim lick rate'] / (exp3['second stim no lick rate']+exp3['second stim lick rate']+.000001)).array
save_name = 'Experimental Population Across Days behavior activity'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2)


title = 'MAN+'
data1 = (eng_pos_exp['second stim lick rate'] / (eng_pos_exp['second stim no lick rate']+eng_pos_exp['second stim lick rate']+.000001)).array
data2 = (eng_pos_exp2['second stim lick rate'] / (eng_pos_exp2['second stim no lick rate']+eng_pos_exp2['second stim lick rate']+.000001)).array
data3 = (eng_pos_exp3['second stim lick rate'] / (eng_pos_exp3['second stim no lick rate']+eng_pos_exp3['second stim lick rate']+.000001)).array
save_name = 'Experimental Engram+ Across Days behavior activity'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2)


title = 'MAN-'
data1 = (eng_neg_exp['second stim lick rate'] / (eng_neg_exp['second stim no lick rate']+eng_neg_exp['second stim lick rate']+.000001)).array
data2 = (eng_neg_exp2['second stim lick rate'] / (eng_neg_exp2['second stim no lick rate']+eng_neg_exp2['second stim lick rate']+.000001)).array
data3 = (eng_neg_exp3['second stim lick rate'] / (eng_neg_exp3['second stim no lick rate']+eng_neg_exp3['second stim lick rate']+.000001)).array
save_name = 'Experimental Engram- Across Days behavior activity'
#agg.boxplot_3(data1, data2, data3, labels, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2, color3=color3)
agg.boxplot_2_days(data1, np.append(data2,data3), labels1, title, ylabel, save_name, save_path, save_figure, is_violin=is_violin, high=high, low=low, color1=color1, color2=color2)


#plt.close('all')


