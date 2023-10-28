#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:48:29 2023

@author: yanshi
"""
import os
import numpy as np
import pandas as pd
import json
from tools.utils import *
f = open('config.json', 'r')
content = f.read()
config = json.loads(content)

atlas_dir=config['atlas_dir']
results_dir=config["results_dir"]
index=config["index"]
dataset=config["dataset"]
data_dir=config["data_dir"][dataset]
subject_dir=config["subject_dir"][dataset]
if os.path.exists(subject_dir) is False: os.makedirs(subject_dir)
subject_file_name=config["subject_file_name"][dataset]
raw_sub_list_path=config["raw_subject_file_path"][dataset]
template_mni_2mm_name=config["atlas_file"]["template_mni_2mm"]
fsl_mask=config["cerebellum_graymatter_mask"]["fsl_mask"]
#template_mni_2mm=os.path.join(atlas_dir,template_mni_2mm)
fsl_mask_type=config["fsl_mask_type"]
cb_mask_name=config["atlas_file"]["cb_mask_mni_2mm"][fsl_mask_type]
mask_threshold=config["mask_threshold"]
dscalar_template_file=config["atlas_file"]["dscalar_template_file"]
dtseries_template_file=config["atlas_file"]["dtseries_template_file"]
sub='0'
#%%
#subject info dir
data_sub=os.path.join(subject_dir,"sub.csv")
data_loss_sub=os.path.join(subject_dir,"sub_loss_data.csv")
sub_num=pd.read_csv(data_sub, header=0).shape[0]
#dataset dir
dataset_dir = os.path.join(results_dir, index , dataset)
if os.path.exists(dataset_dir) is False: os.makedirs(dataset_dir)
# ============
save_dir_individual = os.path.join(dataset_dir, 'individual_voxel')
save_dir_2mm = os.path.join(save_dir_individual, 't1w_t2w')
if os.path.exists(save_dir_2mm) is False: os.makedirs(save_dir_2mm)
# raw data path
t1w_brain_path = os.path.join(data_dir,'{}', 'MNINonLinear', 'T1w_restore_brain.nii.gz')
t2w_brain_path = os.path.join(data_dir,'{}', 'MNINonLinear', 'T2w_restore_brain.nii.gz')
# restore 2mm data path
t1w_restore_name='T1w_restore_brain_2_'+'{}'+'.nii.gz'
t1w_brain_2_path = os.path.join(save_dir_2mm,t1w_restore_name)
t2w_restore_name='T2w_restore_brain_2_'+'{}'+'.nii.gz'
t2w_brain_2_path = os.path.join(save_dir_2mm,t2w_restore_name) 
#template_mni_2mm_path
template_mni_2mm_path=os.path.join(atlas_dir,template_mni_2mm_name)
# t1/t2 myeline content path
myeline_file_name=f'{index}_'+'{}'+'_2mm.nii.gz'
myeline_file_path=os.path.join(save_dir_individual,myeline_file_name)
# cb mask path
cb_mask_path=os.path.join(atlas_dir, cb_mask_name)
# masked t1/t2 myeline content path
masked_myeline_file_name=f'{index}_'+'{}'+f'_{fsl_mask_type}_{mask_threshold}_voxel_2mm.nii.gz'
masked_myeline_file_path=os.path.join(save_dir_individual,masked_myeline_file_name)
# data of all subjects' masked myeline
save_path_fsl_voxel_sublist = os.path.join(dataset_dir, f'{index}_{fsl_mask_type}_voxel_sub')
save_path_fsl_voxel = os.path.join(dataset_dir, f'{index}_{fsl_mask_type}_{mask_threshold}_voxel.nii.gz')
#surface masked myeline content path
surface_dscalar_file=f'{index}_'+'{}'+f'_{fsl_mask_type}_{mask_threshold}_surface_2mm.dscalar.nii' 
surface_file=f'{index}_'+'{}'+f'_{fsl_mask_type}_{mask_threshold}_surface_2mm.nii'
surface_dscalar_path=os.path.join(save_dir_individual,surface_dscalar_file)
surface_path=os.path.join(save_dir_individual,surface_file)
#dscalar_template_file
dscalar_template_file_path=os.path.join(atlas_dir,dscalar_template_file)
dtseries_template_file_path=os.path.join(atlas_dir,dtseries_template_file)
scalar_header=CiftiReader(dscalar_template_file_path).header
a_ciftifile = CiftiReader(dtseries_template_file_path)
#surface_mask
use_threshold=config['atlas_file']['surface_cb_mask_mni_2mm'][fsl_mask_type]["use_threshold"]
if fsl_mask_type in config["atlas_file"]["surface_cb_mask_mni_2mm"]:
    if config['atlas_file']['surface_cb_mask_mni_2mm'][fsl_mask_type]["use_threshold"]=="yes":
        surface_mask_name=config['atlas_file']['surface_cb_mask_mni_2mm'][fsl_mask_type]['nii_file'].format(mask_threshold)
        surface_dscalar_mask_name=config['atlas_file']['surface_cb_mask_mni_2mm'][fsl_mask_type]['dscalar_nii_file'].format(mask_threshold)
    else:
        surface_mask_name=config['atlas_file']['surface_cb_mask_mni_2mm'][fsl_mask_type]['nii_file']
        surface_dscalar_mask_name=config['atlas_file']['surface_cb_mask_mni_2mm'][fsl_mask_type]['dscalar_nii_file']
else:
    pass
surface_mask_path=os.path.join(atlas_dir,surface_mask_name)
surface_dscalar_mask_path=os.path.join(atlas_dir,surface_dscalar_mask_name)
if os.path.exists(surface_mask_path) and os.path.exists(surface_dscalar_mask_path):
    pass
else:
    
    get_surface_mask(a_ciftifile,cb_mask_path, surface_dscalar_mask_path, surface_mask_path, use_threshold, mask_threshold,scalar_header)
    
    
#pca data

pca_source_data=config["pca_source_data"].format(index,fsl_mask_type,mask_threshold,sub_num)
pca_source_data_path=os.path.join(dataset_dir,pca_source_data)
surface_mask_indeces=config['atlas_file']['surface_cb_mask_mni_2mm'][fsl_mask_type]['surface_mask_indeces'].format(fsl_mask_type,mask_threshold)
surface_mask_indeces_path=os.path.join(atlas_dir,surface_mask_indeces)

pca_features=config['pca_feature']
n_components=config["n_components"]
pca_feature_name="".join(pca_features)
pca_dir = os.path.join(results_dir,pca_feature_name, dataset,f"{fsl_mask_type}_{mask_threshold}_pca")
if os.path.exists(pca_dir) is False: os.makedirs(pca_dir)
main_component_dscalar_file_name=config["main_component_dscalar_file_name"]
main_component_file_name=config["main_component_file_name"]
sub_weight=config["sub_weight"]
main_component_dscalar_file=os.path.join(pca_dir,main_component_dscalar_file_name)
main_component_file=os.path.join(pca_dir,main_component_file_name)
sub_weight=os.path.join(pca_dir,sub_weight)
softmax_main_component_dscalar_file_name=config["softmax_main_component_dscalar_file_name"]
softmax_main_component_dscalar_file=os.path.join(pca_dir,softmax_main_component_dscalar_file_name)
softmax_main_component_file_name=config['softmax_main_component_file_name']
softmax_main_component_file=os.path.join(pca_dir,softmax_main_component_file_name)

