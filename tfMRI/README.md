# Preprocessing and Analysis codes for task fMRI part
## Preprocess procedure
The MRI data were first converted into the Neuroimaging Informatics Technology Initiative (NIfTI) format and then organized into the Brain Imaging Data Structure (BIDS) using HeuDiConv (https://github.com/nipy/heudiconv). Then fMRIprep 20.2.1 were used to perform volume prepocess. Detailed information on fMRIprep pipelines can be found in the online documentation of the fMRIPrep(https://fmriprep.org). Then, all the preprocessed individual fMRI data were registered onto the 32k fsLR space using the Ciftify toolbox.

### Volume-based process
**code: ./volume_preprocess/volume_preprocess.sh**

The data2bids.py helps you to reorganize the data structure to BIDS. You have to first prepare the scan_info.xlsx to fit for your experiment protocol and design. 
fMRIprep were performed using docker and detailed usage notes are available in codes, please read carefully and modify variables to satisfy your customed environment.

### Surface-based process
**code: ./surface_preprocess/surface_preprocess.sh**

The cifify_recon_all function was used to register and resample individual surfaces to 32k standard fsLR surfaces via surface-based alignment. The ciftfy_subject_fmri function was then used to project functional MRI data onto the fsLR surface. 
In most circumstances the only necessary operation it to change the path to dataset, other optional settings are explained by annotations.

## Analysis tools
We integrate some useful tools/algorithms which are frequently carried out in task fMRI analysis. 

### Encoding model

**code: ./Encoding.py**

Using stimulus visual and semantic features to predict brain response when viewing these images.

Execute the code in the following order:
encoding_stimuli_preparation.py -- prepare stimulus csv
encoding_feature_extraction.py -- extract different features on the images
encoding_brain_response_preparation.py -- prepare brain response on the images
encoding_main.py -- use features and brain response to fit model

### Multi-voxel pattern analysis (MVPA)

**code: ./MVPA.py**

MVPA is considered as a supervised classification problem where a classifier attempts to capture the relationships between spatial pattern of fMRI activity and experimental conditions. (Mahmoudi et al., 2012).

### Representation similarity analysis (RSA)

**code: ./RSA.py**

A representational similarity analysis (RSA) was conducted to validate that multi-voxel activity patterns from the data represent a rich semantic structure of experimental conditions. 

Specifically, the representational dissimilarity matrix (RDM) of the experimental conditions was constructed by computing the Pearson correlation between the multi-voxel activity patterns from each conditions. 

### Representation dimension reduction

**code: ./RDR.py**

Representation dimension reduction was implemented to reduce the high dimensional voxel space into the low dimensional representation space. In this process, several gradients map were usually observed to inspect the functional organization of the cerebral cortex.


### Inter-subject correlation(ISC)

**code: ./ISC.py**

An inter-subject correlation (ISC) analysis was performed to validate that the data can reveal consistent action category-selective response profiles across participants. 

Here, the ISC is measured for each participant by calculating the Pearson correlation between her/his category-specific response profiles (i.e., beta series) with the averaged category-specific response profiles from the remaining n-1 participants.

### Contrast-to-noise ratio (CNR)

**code: ./CNR.py**

A contrast-to-noise ratio (CNR) analysis was performed to check if the stimulus can induce desired signal changes in each vertex across the cortical surface. 

The CNR was calculated as the averaged beta values across all stimuli divided by the temporal standard deviation of the residual time series from GLM models. 

## Utils

### Input and output 

**code: ./io.py**

We prepare several input and output function to transform neuroimaging files into matrix for further modeling. 


## Reference


