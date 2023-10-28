#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:19:58 2023

@author: yanshi
"""


# %%
import os
#import cb_tools 
from tqdm import tqdm
#import utils
from tools.utils import *
#from utils import full_volume2surface
import config
import nibabel as nib
import numpy as np
import subprocess
import pandas as pd
data_sub=config.data_sub
sublist = pd.read_csv(data_sub, header=0)
masked_myeline_file_path=config.masked_myeline_file_path
config_dscalar_output_path=config.surface_dscalar_path
config_output_path=config.surface_path
dscalar_template_file_path=config.dscalar_template_file_path
a_cifti_file = CiftiReader(dscalar_template_file_path)
scalar_header=config.scalar_header
# %% ====================================================

with tqdm(total=len(sublist)) as pbar:
        # test version:
        #for sub in ['996782','994273']:
        #for sub in sublist['Sub']:
        for i,row in sublist.iterrows():
            sub=str(row[0])
            input_path=masked_myeline_file_path.format(sub)
            #dscalar_output_path=os.path.join(individual_path,dscalar_output_file)
            dscalar_output_path=config_dscalar_output_path.format(sub)
            output_path=config_output_path.format(sub)
            #output_path=os.path.join(individual_path,output_file)
            if os.path.exists(dscalar_output_path):
                pass
            else:
                surface_array=full_volume2surface(a_cifti_file=a_cifti_file,input_path=input_path,dscalar_output_path=dscalar_output_path,output_path=output_path,scalar_header=scalar_header)
                nib.save(nib.Cifti2Image(surface_array,scalar_header),dscalar_output_path)
            string='wb_command -cifti-separate {} COLUMN -volume-all {}'.format(dscalar_output_path,output_path)
            if os.path.exists(output_path):
                pass
            else:
                subprocess.run(string,shell=True)
            pbar.update(1)
