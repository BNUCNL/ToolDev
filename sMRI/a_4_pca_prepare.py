#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:31:11 2023

@author: yanshi
"""
import config
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
import os
import pickle
# %%
#config
data_sub=config.data_sub
sublist = pd.read_csv(data_sub, header=0)
data_sub=config.data_sub
surface_path=config.surface_path
surface_dscalar_path=config.surface_dscalar_path
pca_source_data_path=config.pca_source_data_path
# %%
#mask
surface_dscalar_mask_path=config.surface_dscalar_mask_path
surface_dscalar_mask = nib.load(surface_dscalar_mask_path).get_fdata()
mask_indeces=np.argwhere(surface_dscalar_mask>=1)
mask_indeces=mask_indeces.tolist()
df=pd.DataFrame(mask_indeces,columns=['val1','val2'])
mask_indeces=df.drop('val1',axis=1).values.tolist()
surface_mask_indeces_path=config.surface_mask_indeces_path
# %%
#save mask indeces (surface)
with open(surface_mask_indeces_path, 'wb') as f:
    pickle.dump(mask_indeces, f)
#get mask indeces from dscalar file (input) data
if os.path.exists(pca_source_data_path):
    pass
else:
    with tqdm(total=len(sublist)) as pbar:
        for i,row in sublist.iterrows():
            sub=str(row[0])
            sub_surface_dscalar_path=surface_dscalar_path.format(sub)
            sub_surface_dscalar_data=nib.load(sub_surface_dscalar_path).get_fdata()
            masked_sub_surface_dscalar_data=np.take(sub_surface_dscalar_data,mask_indeces)
            if i == 0:
                pca_source_data=masked_sub_surface_dscalar_data
            else:
                pca_source_data=np.concatenate((pca_source_data,masked_sub_surface_dscalar_data),axis=1)
            pbar.update(1)
        print("concat down")
    np.save(pca_source_data_path,pca_source_data)


