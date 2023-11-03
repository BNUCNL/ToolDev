import os
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from os.path import join as pjoin
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import make_pipeline, Pipeline
from torch import nn
from torch.utils.data import random_split
# personal function
from model_utils import nested_cv, gen_param_grid, class_sample, voxel_selection, across_sub_cv,\
                        Dataset, compute_acc, train, plot_training_curve, gen_pipe
# from brain_inspired_nn import VisualNet, VisualNet_simple, VisualNet_fully_connected, \
#                               HierarchyNet_merge_loss, HierarchyNet_merge_stream, HierarchyNet_merge_stream_PCA

#%% load data and label
main_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/'
beta_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/beta'

# for sub_id in [1,4,5,6,8,9,10]:
#     sub_name = 'sub-{:02d}'.format(sub_id)
#     sub_core_name = 'sub-core{:02d}'.format(sub_id)
# sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub-') and int(i[-2:])<=10]) 
sub_names = ['sub-01-30']
decoding_method = 'sklearn'
voxel_selection_method = 'stability'
group_level = 'sub'
test = 'stable'
ensemble = False
sklearn_model = 'lda'

#%%
# for group_level in ['sub']:
for sub_name in sub_names:
    # data = np.load(pjoin(beta_path, sub_name, f'{sub_name}_imagenet-beta.npy'))
    # df_label = pd.read_csv(pjoin(beta_path, sub_name, f'{sub_name}_imagenet-label.csv'))
    # label_selected = np.load(pjoin(main_path, 'imagenet_decoding', 'data', 'imagenet_10classes_self_defined_equal.npy'))
    out_path = pjoin(main_path, 'imagenet_decoding', 'results')
    
    data = np.load(pjoin(main_path, 'imagenet_decoding', 'data', f'{sub_name}_imagenet-beta.npy'))
    label_raw = np.load(pjoin(main_path, 'imagenet_decoding', 'data', f'{sub_name}_imagenet-label-num_prior.npy'))
    # labels = ['food', 'bird', 'reptile', 'canine', 'invertebrate', 'conveyance', 'device', 'container', 'covering', 'structure']
    # # sort the classes by 1000 categories and transform label
    # class_id = (df_label['class_id'].to_numpy() - 1).astype(int)
    # labels = ['human', 'canine', 'bird', 'snake', 'insect', 'food', 'house', 'car', 'ship', 'musical_instruments']
    # run_idx = np.tile(np.repeat(np.linspace(1, 10, 10, dtype=int), 100), 4)
    # sess_idx = np.repeat(np.linspace(1, 4, 4, dtype=int), 1000)
    # class_id_self_defined = label_selected[class_id]
    # selected_loc = class_id_self_defined != 0
    # # mask data
    # label = class_id_self_defined[selected_loc]
    # run_idx = run_idx[selected_loc]
    # data = data[selected_loc, :]

    # # mask to obtain the visual area data
    # visual_area_voxel = np.load(pjoin(main_path, 'visual_network_idx.npy'))
    # data = data[:, :visual_area_voxel.shape[0]]
    # data = data[:, visual_area_voxel]

    # data, label and run_idx
    label = label_raw[:, 0]
    run_idx = label_raw[:, 1]
    sess_idx = label_raw[:, 2]
    sub_idx = label_raw[:, 3]
    run_idx = run_idx % 10

    groups = label_raw[:, 1:] #eval(f'{group_level}_idx') 
    # sample selection and voxel selection
    data, label, groups = class_sample(data, label, groups)
    print(f'Select {data.shape[0]} samples')
    if voxel_selection_method == 'stability':
        voxel_select_percentage = 25
        n_voxel_select = int(data.shape[1]*(voxel_select_percentage/100))
        stability_score_path = pjoin(main_path, 'imagenet_decoding', 'voxel', 
                                      f'{sub_name}_imagenet-stability_score.npy')
        if not os.path.exists(stability_score_path):
            stability_score = voxel_selection(data, label, groups)
            np.save(stability_score_path, stability_score)
        else:
            stability_score = np.load(stability_score_path)
        select_loc = np.argsort(stability_score)[:n_voxel_select]
        data = data[:, select_loc]
    else:
        roi_path = pjoin(main_path, 'imagenet_decoding', 'voxel', f'{voxel_selection_method}_idx.npy')
        if os.path.exists(roi_path):
            roi_idx = np.load(roi_path)
            data = data[:, roi_idx]
    print(data.shape)
        
    #%% Decoding Main Part
    #======================== For Sklearn Classifier =======================
    if decoding_method == 'sklearn':
        for group_level in ['sess']:
            for sklearn_model in ['lda', 'logistic']:
                info = pd.DataFrame(columns=['single', 'mean'])
                # make model and pipeline
                if ensemble:
                    param_grid = None
                    pipe = gen_voting(sklearn_model, voting='soft', weights=[2, 1, 1])
                else:
                    param_grid = gen_param_grid(sklearn_model)    
                    pipe = gen_pipe(sklearn_model, voxel_selection_method)

                loop_time = 1
                for loop_idx in range(loop_time):
                    # sample data
                    # define nested cv
                    # outer_scores_single, outer_scores_mean, best_params, confusion = nested_cv(data, label, groups, class_name=labels,
                    #                                                                 param_grid=param_grid, Classifier=pipe, 
                    #                                                                 grid_search=False, k=1, mean_times=None,
                    #                                                                 groupby='normal_fold', postprocess=False, 
                    #                                                                 )
                    # define cross_sub cv
                    outer_scores_single, outer_scores_mean = across_sub_cv(data, label, groups, Classifier=pipe, 
                                                                        groupby=group_level, mode='across', test=test, n_bootstrap=1, 
                                                                        postprocess=False, sklearn_model=sklearn_model)
                                                                                    
                    print("Cross-validation scores in single trial: ", outer_scores_single)
                    print("Mean cross-validation score in single trial: ", np.array(outer_scores_single).mean())
                    
                    print("Cross-validation scores in mean pattern: ", outer_scores_mean)
                    print("Mean cross-validation score in mean pattern: ", np.array(outer_scores_mean).mean())
                    # print(best_params)
                    info.loc[loop_idx, ['single', 'mean']] = [np.array(outer_scores_single).mean(), 
                                                            np.array(outer_scores_mean).mean()]
                    print(f'Finish loop {loop_idx} in {sub_name} in {sklearn_model}')
                    
                info.loc[loop_idx+1, ['single', 'mean']] = [info.iloc[:,0].mean(), info.iloc[:,1].mean()]
                info.loc[loop_idx+2, ['single', 'mean']] = [info.iloc[:,0].std(), info.iloc[:,1].std()]
                info.to_csv(f'{out_path}/acc/final_report/{sub_name}-{test}-group_{group_level}-{sklearn_model}-same_sample-{voxel_selection_method}.csv', 
                            index=False)
    #========================== For Neural Network ===================
    elif decoding_method == 'nn':
        # load voxel info
        v1_idx = np.load(pjoin(main_path, 'imagenet_decoding', 'voxel', 'V1_idx.npy'))
        v2_idx = np.load(pjoin(main_path, 'imagenet_decoding', 'voxel', 'V2_idx.npy'))
        v4_idx = np.load(pjoin(main_path, 'imagenet_decoding', 'voxel', 'V4_idx.npy'))
        hvc = np.load(pjoin(main_path, 'imagenet_decoding', 'voxel', 'HVC_idx.npy'))
        ROIs = {'v1':v1_idx, 'v2':v2_idx, 'v4':v4_idx, 'hvc':hvc}
        # define params
        lr = 0.001
        n_epoch = 30
        train_percentage = 0.8
        batch_size = 32
        weight_decay = 1e-5
        p = 0.5
        augment_size = 1000
        n_components = 16
        verbose = False # if True the code will show the loss info in each batch
        train_size = int(data.shape[0] * train_percentage)
        val_size = data.shape[0] - train_size
        # train model
        loop_time = 1
        info = pd.DataFrame(columns=['train_acc', 'val_acc'])
        for loop_idx in range(loop_time):
            # define model and make dataset
            # model = VisualNet(p, selected_voxel=stability_idx)
            model = HierarchyNet_merge_stream_PCA(p, n_hidden=n_components)
            dataset = Dataset(data, label)
            train_set, val_set = random_split(dataset, [train_size, val_size])
            # augmentation on training set
            train_set = Dataset(data=train_set.dataset.data[train_set.indices],
                                labels=train_set.dataset.labels[train_set.indices])
            val_set = Dataset(data=val_set.dataset.data[val_set.indices],
                              labels=val_set.dataset.labels[val_set.indices])
            # train_set.augment(augment_size)
            # PCA on dataset
            train_set.pca_in_ROI(ROIs, n_components)
            val_set.pca_in_ROI(ROIs, n_components)
            # train model
            model_params, train_acc, train_loss, val_acc, val_loss = \
                train(model, train_set, val_set, batch_size, n_epoch, lr, weight_decay)
            info.loc[loop_idx, ['train_acc', 'val_acc']] = [np.array(train_acc).max(),
                                                            np.array(val_acc).max()]   
            print(f'Finish loop {loop_idx+1}\n')
        info.loc[loop_idx+1, ['train_acc', 'val_acc']] = [info.iloc[:,0].mean(), info.iloc[:,1].mean()]
        info.loc[loop_idx+2, ['train_acc', 'val_acc']] = [info.iloc[:,0].std(), info.iloc[:,1].std()]
        info.to_csv(f'{out_path}/acc/nn/{sub_name}-merge_stream_pca_component_{n_components}.csv', index=False)
        # save and plot info
        plot_training_curve(n_epoch, train_acc, train_loss, val_acc, val_loss, 
                            flag=sub_name+f'_merge_stream_pca_component_{n_components}')
        # torch.save(model_params, pjoin(out_path, 'visual_nn.pkl'))
        # _merge_stream_pca_component_{n_components}
    

