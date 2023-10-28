#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:16:41 2023

@author: yanshi
"""

# %%
# config
import config
pca_source_data_path=config.pca_source_data_path
surface_mask_indeces_path=config.surface_mask_indeces_path
pca_features=config.pca_features
n_components=config.n_components
pca_feature_name=config.pca_feature_name
main_component_dscalar_file=config.main_component_dscalar_file
main_component_file=config.main_component_file
sub_weight_file=config.sub_weight
pca_dir=config.pca_dir
scalar_header=config.scalar_header
#softmax_main_component_dscalar_file=config.softmax_main_component_dscalar_file
#softmax_main_component_file=config.softmax_main_component_file
# %%
#calculate pca
import numpy as np
from sklearn.decomposition import PCA
if n_components==0:
    n_components=None
elif type(n_components) is int:
    pass
elif type(n_components) is str:
    n_components='mle'
pca_source_data=np.load(pca_source_data_path)
pca=PCA(n_components=n_components,copy=True)
main_component=pca.fit_transform(pca_source_data)
sub_weight_data=pca.components_.T
explained_variance_ratio_list=pca.explained_variance_ratio_
#%% vis
from tools.vis import draw_line_chart
draw_line_chart(explained_variance_ratio_list)
#%%
#save file
import nibabel as nib
import os
import subprocess
from tqdm import tqdm
import pickle
from scipy.special import softmax 

if os.path.exists(sub_weight_file):
    pass
else:
    #print(f)
    np.save(sub_weight_file,sub_weight_data)
    
    
    
    
with tqdm(total=main_component.shape[1]) as pbar:
    for i in range(main_component.shape[1]):
        a_component=main_component[:,i]
        #softmax_a_component=softmax(a_component)
        explained_variance_ratio=explained_variance_ratio_list[i]
        #features_variance_paiming_.dscalar.nii"
        component_file=main_component_file.format(pca_feature_name,(i+1),explained_variance_ratio)
        component_dscalar_file=main_component_dscalar_file.format(pca_feature_name,(i+1),explained_variance_ratio)
        #softmax_component_dscalar_file=softmax_main_component_dscalar_file.format(pca_feature_name,(i+1),explained_variance_ratio)
        #softmax_component_file=softmax_main_component_file.format(pca_feature_name,(i+1),explained_variance_ratio)
        
        if os.path.exists(component_dscalar_file):
        #if os.path.exists(component_dscalar_file) and os.path.exists(softmax_component_dscalar_file):
            pass
        else:
            #softmax_surface_gradient=np.full((1,91282),0.0)
            surface_gradient=np.full((1,91282),0.0)
            with open(surface_mask_indeces_path, 'rb') as f:
                surface_mask_indeces = pickle.load(f)
            for index,voxel in enumerate(surface_mask_indeces):
                surface_gradient[0,voxel[0]]=a_component[index]
                #softmax_surface_gradient[0,voxel[0]]=softmax_a_component[index]
            nib.save(nib.Cifti2Image(surface_gradient,scalar_header),component_dscalar_file)
            #nib.save(nib.Cifti2Image(softmax_surface_gradient,scalar_header),softmax_component_dscalar_file)
        if os.path.exists(component_file):
            pass
        else:
            string='wb_command -cifti-separate {} COLUMN -volume-all {}'.format(component_dscalar_file,component_file)
            subprocess.run(string,shell=True)
            '''
        if os.path.exists(softmax_component_file):
            pass
        else:
            string='wb_command -cifti-separate {} COLUMN -volume-all {}'.format(softmax_component_dscalar_file,softmax_component_file)
            subprocess.run(string,shell=True)
            '''
        pbar.update(1)