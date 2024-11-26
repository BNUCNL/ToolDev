import os
import config
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from scipy.stats import zscore


def prepare_NOD_response(sub_name, clean_code='hp128_s4'):
 
    # prepare unique params
    ciftify_path = f'{dataset_path}/derivatives/ciftify'
    num_ses, num_run, num_trial = 4, 10, 100 
    vox_num = 59412

    # load stimulus csv
    label_file = pjoin(support_path, f'{sub_name}_imagenet-label.csv')
    label_filtered_file = pjoin(support_path, f'{sub_name}_imagenet-label-filtered.csv')
    df_img_name = pd.read_csv(label_file)
    df_filtered = pd.read_csv(label_filtered_file)
    # prepare resort index
    image_id = df_filtered['image_id'].to_list()
    image_sum = df_img_name['image_id'].to_list()
    image_order = [image_sum.index(img) for img in image_id]

    # define beta path
    beta_sub_path = pjoin(support_path, f'{sub_name}_imagenet-beta_{clean_code}_filtered_ridge.npy')
    if not os.path.exists(beta_sub_path):
        # extract from dscalar.nii
        beta_sub = np.zeros((num_ses, num_run*num_trial, vox_num))
        for i_ses in range(num_ses):
            for i_run in range(num_run):
                run_name = f'ses-imagenet{i_ses+1:02d}_task-imagenet_run-{i_run+1}'
                beta_data_path = pjoin(ciftify_path, sub_name, 'results', run_name, f'{run_name}_beta.dscalar.nii')
                beta_sub[i_ses, i_run*num_trial : (i_run + 1)*num_trial, :] = np.asarray(nib.load(beta_data_path).get_fdata())
        # save session beta in ./supportfiles
        beta_sub = beta_sub.reshape((-1,beta_sub.shape[-1]))
        beta_sub = beta_sub[image_order]
        np.save(beta_sub_path, beta_sub)

# Set up the argument parser
parser = argparse.ArgumentParser(description='Model fitting for fMRI data.')
parser.add_argument('--sub_name', type=str, nargs='+', default=['sub-04'], 
                    help='Subject names', required=True)

# Parse the arguments
args = parser.parse_args()

# parameters setting for user input
sub_names = args.sub_name

# parameters setting
support_path = config.support_path
dataset_path = config.dataset_path

# loop to prepare brain response
for sub_name in sub_names:
    prepare_NOD_response(sub_name)
