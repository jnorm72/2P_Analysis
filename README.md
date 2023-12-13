# 2P_Analysis
This repository contains Python scripts for analyzing 2P imaging data and combining it with cell label images and behavioral data. 

## Included Files
* **CaImAn_deconv_batch.py** extracts the cellular activity and spatial footprints from an imaging session. Currently takes in raw files and saves the data in an HDF5. Additional visualizations (correlation image, video) are also saved. 
* **hdf5_cleaning.py** uses a custom putative spike deconvolution algorithm with tunable parameters to determine cellular activity. Also allows the user to clean the data by removing cells or time periods that have poor data quality. 
* **cell_label_pipeline.py** incorporates a static image with cell labels to identify the labeled and unlabeled cells. Currently built for raw files that also have a summary image. Works by registering the green images between sessions, subtracting out the bleed through from green into red, and by comparing the red fluorescence in the label session to the cell footprints of the trial.

* **general_utils.py** contains functions required for the above scripts. Simple functions such as finding and loading data types.
* **CaImAn_deconv_batch_utils.py** contains functions required for CaImAn_deconv_batch.py. 
* **cleaning_utils.py** contains functions required for hdf5_cleaning.py. 
* **cell_label_utils.py** contains functions required for cell_label_pipeline.

* **put_spike_deconv.py** contains a function that extracts putative spikes. Used in cleaning_utils.py.
* **find_offset.py** contains a function that finds the offset between two fields of view. Used in cell_label_utils.py.


## Installation and Package Requirements
A CaImAn environment must be installed and activated prior to running CaImAn_deconv_batch.py. 

## Data Requirements
Imaging sessions and cell label sessions should be .raw files with the metadata in an .xml file. The trial imaging session should only be 1 channel, while the cell label session should be 2 channels. 
Behavioral and stimulation data should be in .abf file format. 
