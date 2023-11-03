import os
import config
import argparse
import numpy as np
import pandas as pd
from os.path import join as pjoin


def prepare_stimlus_set(sub_names):
    # define path
    save_path = pjoin(support_path, './supportfiles/sub_stim')
    nifti_path = f'{dataset_path}'
    stim = pd.read_csv(pjoin(support_path, 'nod_stim.csv'), encoding='gbk')
    stim = stim.sort_values(by=['sub', 'sess', 'run'])
    # make path
    if not os.path.exists(support_path):
        os.makedirs(support_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # loop for each subject
    for sub_idx, sub_name in enumerate(sub_names):
        print(sub_name)
        # get annotation info
        image_in_annotation = stim.loc[stim['sub']==sub_name, 'image_name'].to_list()

        label_file = pjoin(support_path, f'{sub_name}_imagenet-label.csv')
        label_filtered_file = pjoin(support_path, f'{sub_name}_imagenet-label-filtered.csv')
        # check whether label exists, if not then generate  
        sub_events_path = pjoin(nifti_path, sub_name)
        df_img_name = []
        # find imagenet task
        imagenet_sess = [_ for _ in os.listdir(sub_events_path) if ('imagenet' in _) and ('05' not in _)]
        imagenet_sess.sort()# Remember to sort list !!!
        # loop sess and run
        for sess in imagenet_sess:
            for run in np.linspace(1,10,10, dtype=int):
                # open ev file
                events_file = pjoin(sub_events_path, sess, 'func',
                                    '{:s}_{:s}_task-imagenet_run-{:02d}_events.tsv'.format(sub_name, sess, run))
                tmp_df = pd.read_csv(events_file, sep="\t")
                tmp_df['image_id'] = [_.split('/')[-1] for _ in tmp_df['stim_file'].to_list()]
                df_img_name.append(tmp_df.loc[:, ['trial_type', 'stim_file', 'image_id']])
        df_img_name = pd.concat(df_img_name)
        df_img_name.columns = ['class_id', 'image_name', 'image_id']
        df_img_name.reset_index(drop=True, inplace=True)
        # load sub label file and select part image
        df_filtered = df_img_name[df_img_name['image_id'].isin(image_in_annotation)]
        # save file
        df_img_name.to_csv(label_file, index=False)
        df_filtered.to_csv(label_filtered_file, index=False)
        print(f'Finish preparing labels for {sub_name}')

        # prepare stimulus set for feature extraction using in DNNBrain
        header = ['type=image\n', f'path={dataset_path}/stimuli/\n', f'title=ImageNet images in {sub_name}\n', 'data=stimID\n']
        # stim files
        sub_stim = pd.read_csv(pjoin(support_path, f'{sub_name}_imagenet-label-filtered.csv'), sep=',')
        # replace file name
        stim_files = '\n'.join(sub_stim['image_name'].tolist())
        with open(f'{save_path}/{sub_name}_imagenet.stim.csv', 'w') as f:
            f.writelines(header)
            f.writelines(stim_files)

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

# loop to prepare stimulus
prepare_stimlus_set(sub_names)