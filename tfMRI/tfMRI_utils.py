import os
import math
import time
import torch
import cifti2
import matplotlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import torch.nn as nn
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.stats import zscore
from scipy.spatial import distance_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dnnbrain.dnn.core import Activation
from scipy import stats
import nibabel as nib
import scipy.io as sio

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, pairwise_distances
from torch.utils.data import DataLoader
from torch import optim

def get_data(sub):
    """
    return response and label
    """   
    main_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/'
    out_path = pjoin(main_path, 'cognitive_state_decoding', 'data')
    return np.load(pjoin(out_path,f'whole_brain/sub-{sub}_imagenet-dtseries.npy')), np.load(pjoin(out_path,f'whole_brain/sub-{sub}_imagenet-label.npy'))

def plot_training_curve(n_epoch, train_acc, train_loss, val_acc, val_loss, flag):
    """
    

    Parameters
    ----------
    n_epoch : TYPE
        DESCRIPTION.
    train_acc : TYPE
        DESCRIPTION.
    train_loss : TYPE
        DESCRIPTION.
    val_acc : TYPE
        DESCRIPTION.
    val_loss : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    font_title = {'family': 'arial', 'weight': 'bold', 'size':14}
    font_other = {'family': 'arial', 'weight': 'bold', 'size':10}
    plt.figure(figsize=(10,4))
    # loss pic
    plt.subplot(1,2,1)
    plt.title("Training Curve")
    plt.plot(range(n_epoch), train_loss, label="Train")
    plt.plot(range(n_epoch), val_loss, label="Validation")
    plt.xticks(fontproperties='arial', weight='bold', size=10)
    plt.yticks(fontproperties='arial', weight='bold', size=10)
    plt.legend(prop=font_other, loc='best')
    
    ax = plt.gca()
    ax.set_xlabel('Iterations', font_other)
    ax.set_ylabel('Loss', font_other)
    ax.set_title('Training Curve', font_title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
       
    # acc pic
    plt.subplot(1,2,2)
    plt.plot(range(n_epoch), train_acc, label="Train")
    plt.plot(range(n_epoch), val_acc, label="Validation")
    plt.plot([-1,n_epoch], [0.1, 0.1], ls='--', color='gray', lw=1.5)
    plt.xticks(fontproperties='arial', weight='bold', size=10)
    plt.yticks(fontproperties='arial', weight='bold', size=10)
    plt.legend(prop=font_other, loc='best')
    
    ax = plt.gca()
    ax.set_xlabel('Iterations', font_other)
    ax.set_ylabel('Training Accuracy', font_other)
    ax.set_title('Training Curve', font_title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.savefig(pjoin('/nfs/z1/zhenlab/BrainImageNet/Analysis_results/imagenet_decoding/results/training_curve', 
                      f'training_curve_{flag}.jpg'))
    plt.close()

def plot_confusion_matrix(confusion, class_name, specify):
    """

    Parameters
    ----------
    specify : str
        Name to specify this plot

    """
    out_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/imagenet_decoding/results/confusion_matrix'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # get confusion
    n_class = confusion.shape[0]
    # visualize
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    cmap = plt.cm.jet
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams.update({'font.size': 12, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
    font = {'family': 'stix', 'weight': 'bold', 'size':14}

    im = plt.imshow(confusion, cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel('Predict label', font)
    ax.set_ylabel('True label', font)
    ax.set_xticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8))
    ax.set_yticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8))
    ax.set_xticklabels(class_name, font, rotation=30, )
    ax.set_yticklabels(class_name, font)
    ax.set_title(f'Confusion matrix {specify}', font)
    plt.tight_layout()
    plt.savefig(pjoin(out_path, f'confusion_{specify}.jpg'))
    plt.close()


def save_classification_report(y_test, y_pred, specify):
    """

    Parameters
    ----------
    y_test : ndarray
        Groundtruth class 
    y_pred : ndarray
        Class predicted by model
    specify : str
        Name to specify this plot

    """
    out_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/imagenet_decoding/results/classification_report'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # generate report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(pjoin(out_path, f'classification_report_{specify}.csv'))

def top_k_acc(X_probs, y_test, k, class_name, verbose=False):
    """
    Accuracy on top k

    Parameters
    ----------
    X_probs : ndarray
        DESCRIPTION.
    y_test : ndarray
        GroundTruth
    k : int
        DESCRIPTION.
    class : array
        Class label names

    Returns
    -------
    acc_top_k : TYPE
        DESCRIPTION.

    """
    # top k
    best_n = np.argsort(X_probs, axis=1)[:, -k:]
    y_top_k = class_name[best_n]
    acc_top_k = np.mean(np.array([1 if y_test[n] in y_top_k[n] else 0 for n in range(y_test.shape[0])]))
    if verbose:
        y_top_k = y_top_k.reshape((10, 10))
        y_test = y_test.reshape((10, 10))
        for sub in range(10):
            acc_sub = np.mean(np.array([1 if y_test[sub, n] == y_top_k[sub, n] \
                                        else 0 for n in range(y_test[sub, :].shape[0])]))
            print('sub %02d acc %.2f\n'%(sub+1, acc_sub))
    return  acc_top_k

def voxel_selection(X, y, groups, method='stability'):
    """
    Select voxels based on pattern stability

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    groups : TYPE
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.
    percentage : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # select voxels highly responsive if desired
    # if method=='active':
    #     voxel_pattern = np.max(X, axis=0)
    #     select_loc = np.argsort(-voxel_pattern)[:n_voxel_select]
    #     X = X[:, select_loc]
    if method=='stability':
        # find groups that all class appears
        run_intere = []
        for run_label in np.unique(groups):
            run_loc = groups == run_label
            label_run = y[run_loc]
            if np.unique(label_run).shape[0] == np.unique(y).shape[0]:
                run_intere.append(run_label)
        # compute stability score
        stability_score = np.zeros((X.shape[1]))
        for voxel_idx in range(X.shape[1]):
            data_voxel = X[:, voxel_idx]
            # define pattern for each voxel
            # find runs that have 10 class
            voxel_pattern = np.zeros((len(run_intere), np.unique(y).shape[0]))
            for run_loop_idx,run_label in enumerate(run_intere):
                run_loc = groups == run_label
                data_run = data_voxel[run_loc]
                label_run = y[run_loc]
                for class_loop_idx,class_label in enumerate(np.unique(y)):
                    class_loc = label_run == class_label
                    data_class = data_run[class_loc]
                    voxel_pattern[run_loop_idx, class_loop_idx] = np.mean(data_class, axis=0)
            print(f'Finish computing {voxel_idx} voxels in stability voxel selection')
            # compute stability score
            corr_matrix = pairwise_distances(voxel_pattern, metric='correlation')
            valid_value = np.triu(corr_matrix, 1).flatten()
            stability_score[voxel_idx] = np.mean(valid_value[valid_value!=0])
    return stability_score      


def across_sub_cv(X, y, groups, Classifier, groupby, mode='across', test='stable',
                  n_bootstrap=1, k=1, postprocess=False, sklearn_model=None):
    """
    Cross validation cross subjects.
    The first part subjects have scanned multiple session, while the other subjects have only scanned one session
    The train set will contain one session from all subjects, and the test set will only contain the sessions that 
    from the first part subjects

    Parameters
    ----------
    X : array-like of shape(n_samples, n_feautre)
        Training vectors
    y : array-like of shape(n_samples,)
        Target values(class label in classification)
    groups : ndarray
        Groups to constrain the cross-validation. We will use sess_idx in Gallent cv
    Classifier : sklearn classifier object
        Sklearn classifier.    
    """
    # define containers and params
    outer_scores_mean = []
    outer_scores_single = []
    class_name = np.unique(y)
    confusion = np.zeros((10, class_name.shape[0], class_name.shape[0]))
    sess_idx = groups[:, 0]
    sub_idx = groups[:, 1]
    
    for bootstrap_index in range(n_bootstrap):
        # prepare train set and test set sess index
        train_set_idx = []
        test_set_idx = []
        if mode == 'stable':
            # generate train set and test set idx
            sub_names_test_part = np.arange(10) 
            for sub_tmp in sub_names_test_part:
                sess_tmp = np.unique(sess_idx[sub_idx == sub_tmp])
                train_tmp = np.random.choice(sess_tmp, 1, replace=False).tolist()
                test_tmp = list(set(sess_tmp).difference(set(train_tmp)))
                train_set_idx.extend(train_tmp)
                test_set_idx.extend(test_tmp)
            # add second part subjects in train set index
            train_set_idx.extend(np.linspace(39, 58, 20, dtype=int))
            # prepare train set and test set sess data
            train_index = [True if x in train_set_idx else False for x in sess_idx]
            test_index = [True if x in test_set_idx else False for x in sess_idx]
        elif mode == 'across':
            if test == 'stable':
                train_set_idx = np.unique(sub_idx)[np.unique(sub_idx) >= 10]
                test_set_idx = np.arange(10) 
            elif test == 'diverse':
                train_set_idx = np.arange(10) 
                test_set_idx = np.unique(sub_idx)[np.unique(sub_idx) >= 10]
            # prepare train set and test set sess data
            train_index = [True if x in train_set_idx else False for x in sub_idx]
            test_index = [True if x in test_set_idx else False for x in sub_idx]
            
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # start computing
        # transform mean pattern 
        X_test_mean = np.zeros((1, X.shape[1]))
        y_test_mean = []
        # define different groups
        group_test = np.unique(eval(f'{groupby}_idx')[test_index])
        group_test_single = eval(f'{groupby}_idx')[test_index]
        # start merging
        idx_all = []
        for idx in np.unique(group_test):
            tmp_X = X_test[group_test_single==idx, :]
            tmp_y = y_test[group_test_single==idx]
            # loop to tranform mean pattern in each class
            for class_idx in np.unique(y):
                # generate mean pattern for class in each run
                if class_idx in tmp_y:
                    class_loc = tmp_y == class_idx
                    pattern = np.mean(tmp_X[class_loc], axis=0)[np.newaxis, :]
                    X_test_mean = np.concatenate((X_test_mean, pattern), axis=0)
                    y_test_mean.append(class_idx)     
                    idx_all.append(idx)
        X_test_mean = np.delete(X_test_mean, 0, axis=0)
        y_test_mean = np.array(y_test_mean)
        # 
        print(y_test.shape)
        print(y_test_mean.shape)
        # fit models
        model = Classifier
        model.fit(X_train, y_train)
        # handle specified situation on svm
        if sklearn_model in ['svm']:
            outer_scores_single.append(model.score(X_test, y_test))
            outer_scores_mean.append(model.score(X_test_mean, y_test_mean))
        else:
            # get topk score
            X_probs_mean = model.predict_proba(X_test_mean)
            X_probs = model.predict_proba(X_test)
            # test score in outer loop
            outer_scores_single.append(top_k_acc(X_probs, y_test, k, class_name))
            outer_scores_mean.append(top_k_acc(X_probs_mean, y_test_mean, k, class_name))
    
        if postprocess:
            # postprocess: including confusion matrix, classification report
            # predict
            y_pred_mean = model.predict(X_test_mean)
            y_pred = model.predict(X_test)
            # plot and save info
            confusion[bootstrap_index-1] = confusion_matrix(y_test_mean, y_pred_mean, normalize='true')
            save_classification_report(y_test_mean, y_pred_mean, f'mean_split{bootstrap_index}')
            save_classification_report(y_test, y_pred, f'single_split{bootstrap_index}')
            
        print(f'Finish bootstrap{bootstrap_index+1}')
        bootstrap_index += 1
        
    if postprocess:
        confusion = np.mean(confusion, axis=0)
        plot_confusion_matrix(confusion, class_name, 'mean_pattern')

    return outer_scores_single, outer_scores_mean
        
        

def nested_cv(X, y, groups, Classifier, class_name=None, param_grid=None, k=1, grid_search=False,
              groupby=None, sess=None, mean_times=None, postprocess=False):
    """
    Nested Cross validation with fMRI fold

    Parameters
    ----------
    X : array-like of shape(n_samples, n_feautre)
        Training vectors
    y : array-like of shape(n_samples,)
        Target values(class label in classification)
    groups : ndarray
        Groups to constrain the cross-validation. We usually use run_idx in fMRI fold.
    Classifier : sklearn classifier object
        Sklearn classifier.
    class_name : list
        Name of the label categories to visualize in confusion matrix
    param_grid : dict
        Parameters info in corresponding classifier.
    k : int
        Top k acc. Different k will have different chance level.
    grid_search : bool
        if True, the cv will start grid searching based on param_grid
    groupby : str, optional
        Define the cross validtion in which groups.
        The choices are normal_fold, fMRI_fold and single_sess
        In normal_fold, the test set will be just randomly selected from the whole dataset.
        In fMRI_fold, the test set will be several runs randomly selected from different session
            or several sessions randomly selected from different subject. It considers the fMRI data structure.
        In single_sess, the train set and test set are using one session data. Note to assign the 
            the sess value if using the single_ses group_by
    sess : int, optional
        Define the session number when groupby is single_sess. The default is None.
    mean_times : int
        Define the mean times of test set sample in a same class.
    postprocess : bool
        if True, it will generate classification report and confusion_matrix.
        Make sure to adjust the path in function plot_confusion_matrix and save_classification_report
    feature_selection : int 
        The percentage of feature selection in active-based voxel selction.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    outer_scores_single : list
        The score in single trial.
    outer_scores_mean : list
        The score in mean pattern.
    best_params : list
        Best params in grid searching.
    X_train
    X_test
    y_train
    y_test

    """
    # define containers
    outer_scores_mean = []
    outer_scores_single = []
    best_params = []

    # change groups 
    # group by sess fold, which contains 4 runs from different sess in each subject
    if groupby[-4:] == 'fold':
        # define cv num
        if groupby == 'normal_fold': 
            n_inner_cv, n_outer_cv = 4, 5
            groups_new = groups
        else:
            if groupby == 'fMRI_fold':
                n_inner_cv, n_outer_cv = 9, 10
            # assign new groups   
            groups_new = np.zeros(groups.shape)
            n_unique = np.unique(groups).shape[0]
            sess_fold = np.arange(n_unique).reshape(int(n_unique/n_outer_cv), n_outer_cv)
            sess_fold = sess_fold + int(groups.min())
            # shuffle sess
            for idx in range(sess_fold.shape[0]):
                np.random.shuffle(sess_fold[idx])
            # generate new fold
            for idx in range(sess_fold.shape[1]):
                target_runs = sess_fold[:, idx]
                fold_loc = np.asarray([True if x in target_runs else False for x in groups])
                groups_new[fold_loc] = idx
    # group by single session, each run is a fold        
    elif groupby == 'single_sess':
        if sess == None:
            raise ValueError('Please assign sess if groupby is single_sess!')
        # define cv num
        n_inner_cv, n_outer_cv = 9, 10
        sess_run = np.arange(10) + (sess-1)*10
        sess_loc = np.asarray([True if x in sess_run else False for x in groups])
        X = X[sess_loc, :]
        y = y[sess_loc]
        groups_new = groups[sess_loc]
        print(X.shape)
        print(f'Nested CV on Sess{sess}')
        
    # start cross validation
    split_index = 1
    class_label = np.unique(y)
    confusion = np.zeros((n_outer_cv, class_label.shape[0], class_label.shape[0]))
    # handle situation for not group cv
    if groupby == None: 
        model = Classifier
        outer_scores_single = cross_val_score(model, X, y, cv=10)
    else:
        # define groupcv
        inner_cv = GroupKFold(n_splits = n_inner_cv)
        outer_cv = GroupKFold(n_splits = n_outer_cv)
        # group cv
        for train_index, test_index in outer_cv.split(X, y, groups=groups_new):
            # split train test
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            groups_cv = groups_new[train_index]
            # in group_run and group_sess, the groups have the same shape but its' content is different
            # as for single sess, the test index should correspond to groups_new
            if groupby == 'single_sess':
                run_test = groups_new[test_index]
            else:
                run_test = groups[test_index]
            # transform mean pattern 
            X_test_mean = np.zeros((1, X.shape[1]))
            y_test_mean = []
            for idx in np.unique(run_test):
                tmp_X = X_test[run_test==idx, :]
                tmp_y = y_test[run_test==idx]
                # loop to tranform mean pattern in each class
                for class_idx in np.unique(y):
                    if mean_times == None:
                        # generate mean pattern for class in each run
                        if class_idx in tmp_y:
                            class_loc = tmp_y == class_idx
                            pattern = np.mean(tmp_X[class_loc], axis=0)[np.newaxis, :]
                            X_test_mean = np.concatenate((X_test_mean, pattern), axis=0)
                            y_test_mean.append(class_idx)
                    else:
                        # generate mean pattern in specified times
                        animacy_loc = tmp_y == class_idx
                        animacy_X = tmp_X[animacy_loc, :]
                        for mean_idx in range(int(animacy_X.shape[0]/mean_times)):
                            pattern = np.mean(animacy_X[mean_idx*mean_times:(mean_idx+1)*mean_times-1], 
                                              axis=0)[np.newaxis, :]
                            X_test_mean = np.concatenate((X_test_mean, pattern), axis=0)
                            y_test_mean.append(class_idx)
            
            X_test_mean = np.delete(X_test_mean, 0, axis=0)
            y_test_mean = np.array(y_test_mean)
            # 
            print(y_test.shape)
            print(y_test_mean.shape)
            # fit grid in inner loop
            if grid_search:
                model = GridSearchCV(Classifier, param_grid, cv=inner_cv, n_jobs=8, verbose=10)
                model.fit(X_train, y_train, groups=groups_cv)
                best_params.append(model.best_params_)
            else:
                model = Classifier
                model.fit(X_train, y_train)
            # handle specified situation on svm
            if param_grid['classifier'][0].__class__.__name__ in ['SVC', 'Lasso'] or groupby == 'single_sess':
                outer_scores_single.append(model.score(X_test, y_test))
                outer_scores_mean.append(model.score(X_test_mean, y_test_mean))
            else:
                # get topk score
                X_probs_mean = model.predict_proba(X_test_mean)
                X_probs = model.predict_proba(X_test)
                # test score in outer loop
                outer_scores_single.append(top_k_acc(X_probs, y_test, k, class_label))
                outer_scores_mean.append(top_k_acc(X_probs_mean, y_test_mean, k, class_label))
        
            if postprocess:
                # postprocess: including confusion matrix, classification report
                # predict
                y_pred_mean = model.predict(X_test_mean)
                y_pred = model.predict(X_test)
                # plot and save info
                confusion[split_index-1] = confusion_matrix(y_test_mean, y_pred_mean, normalize='true')
                save_classification_report(y_test_mean, y_pred_mean, f'mean_split{split_index}')
                save_classification_report(y_test, y_pred, f'single_split{split_index}')
                
            print(f'Finish cv in split{split_index}')
            split_index += 1
            
        if postprocess:
            confusion = np.mean(confusion, axis=0)
            if class_name == None:
                class_name = class_label
            plot_confusion_matrix(confusion, class_name, 'mean_pattern')

    return outer_scores_single, outer_scores_mean, best_params, confusion
        

def class_sample(data, label, groups):
    """
    Make each class has the same sample based on the distance 
    between sample and class mean pattern

    Parameters
    ----------
    data : ndarray
        n_sample x n_feature
    label : ndarray
        DESCRIPTION.
    run_idx : ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # sample class
    n_sample = pd.DataFrame(label).value_counts().min()
    data_sample = np.zeros((1, data.shape[1]))
    if len(groups.shape) == 2:
        groups_sample = np.zeros((1, groups.shape[1]))
    else:
        groups_sample = np.zeros((1, 1))
        groups = groups[:, np.newaxis]
    # loop to sample
    for idx,class_idx in enumerate(np.unique(label)):
        class_loc = label == class_idx 
        class_data = data[class_loc]
        class_mean = np.mean(class_data, axis=0)[np.newaxis, :]
        eucl_distance = distance_matrix(class_data, class_mean).squeeze()
        # random select sample to make each class has the same number
        select_idx = np.argsort(eucl_distance)[:n_sample]
        data_class = data[class_loc, :][select_idx]
        groups_class = groups[class_loc, :][select_idx, :]
        # concatenate on the original array
        data_sample = np.concatenate((data_sample, data_class), axis=0)
        groups_sample = np.concatenate((groups_sample, groups_class), axis=0)
    # prepare final data
    data_sample = np.delete(data_sample, 0, axis=0)
    groups_sample = np.delete(groups_sample, 0, axis=0)
    label_sample = np.repeat(np.unique(label), n_sample)
    groups_sample = groups_sample.squeeze()
    
    return data_sample, label_sample, groups_sample
    
    

def find_outlier(data, label, cont):
    # input: data,  contamination -> outlier ratio
    # output: scatter figure;
    # return: X_pca -> PCA processed data, array of float64
    #         y_pred -> outlier mark, array of int64 (1 & -1)
    
    out_index = []
    
    for class_idx in np.unique(label):
        
        # get class data
        class_label = label == class_idx
        class_data = data[class_label, :]
        class_loc = np.where(class_label==1)[0]
        
        # scaler
        scaler = StandardScaler()
        scaler.fit(class_data)
        X_scaled = scaler.transform(class_data)
        
        # PCA
        pca = PCA(n_components=2)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        
        # EllipticEnvelope to find outlier
        esti = EllipticEnvelope(contamination=cont)
        y_pred = esti.fit(X_pca).predict(X_pca)
       
        # store outlier index
        out_index.extend(class_loc[np.where(y_pred == -1)[0]])
        print(f'Finish finding outlier index in class {class_idx}')

    # return
    return out_index


def gen_param_grid(method):
    
    param_grid = {
                  'svm':    
                   {'classifier': [SVC(max_iter=8000)], 'feature_selection':[SelectPercentile()],
                    'classifier__C': [0.001],
                    'classifier__kernel': ['linear'],
                    'classifier__decision_function_shape': ['ovo'],
                    'feature_selection__percentile': [25],},
                  'logistic':    
                      {'classifier': [LogisticRegression(max_iter=8000)], 
                       'feature_selection':[SelectPercentile()],
                       'classifier__C': [0.001],
                       'classifier__solver': ['liblinear'],
                       'feature_selection__percentile': [25],},
                  'rf':    
                      {'classifier': [RandomForestClassifier()], 'feature_selection':[SelectPercentile()],
                       'classifier__n_estimators': [500, 300, 200, ],
                       'feature_selection__percentile': [25],},
                  'mlp':
                      {'classifier': [MLPClassifier()], 'feature_selection':[SelectPercentile()],
                       'classifier__alpha': [0.01],
                       'classifier__hidden_layer_sizes': [(200,)],
                       'feature_selection__percentile': [25],},
                  'lasso':    
                      {'classifier': [Lasso(max_iter=8000)],
                       'classifier__alpha': [0.001, 0.01, 0.1, 1],}, 
                  'lda':    
                   {'classifier': [LinearDiscriminantAnalysis()], 'feature_selection':[SelectPercentile()],
                    'classifier__solver': ['lsqr'],
                    'classifier__shrinkage': [0.9],
                    'feature_selection__percentile': [20,40,60],},
                  }  

    return param_grid[method]            

def gen_pipe(model, voxel_selection_method):
    """
    Prepare the model params info after grid searching

    Parameters
    ----------
    method : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if voxel_selection_method == 'discrim':
        pipe = {
                'svm':    
                    Pipeline([('feature_selection', SelectPercentile(percentile=25)), 
                              ('classifier', SVC(max_iter=8000, C=0.001, kernel='linear', decision_function_shape='ovo'))]),
                'logistic':    
                    Pipeline([('feature_selection', SelectPercentile(percentile=25)), 
                              ('classifier', LogisticRegression(C=0.001, max_iter=8000, solver='liblinear'))]),
                'lda':    
                    Pipeline([('feature_selection', SelectPercentile(percentile=25)), 
                              ('classifier', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9)),])
               }  
    else:
        pipe = {
                'svm':    
                    Pipeline([('classifier', SVC(max_iter=8000, C=0.001, kernel='linear', decision_function_shape='ovo', verbose=True))]),
                'logistic':    
                    Pipeline([('classifier', LogisticRegression(C=0.001, max_iter=8000, solver='liblinear', verbose=True))]),
                'lda':    
                    Pipeline([('classifier', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9)),])
               }  
        
        ### best params after grid searching ###
        # LogisticRegression(C=0.001, max_iter=8000, solver='liblinear')
        # MLPClassifier(hidden_layer_sizes=100, alpha=0.01)
        # SVC(max_iter=8000, C=0.001, kernel='linear', decision_function_shape='ovo')
        # RandomForestClassifier(n_estimators=500)
        # Lasso(alpha=0.01)
        # LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.9)
            
    return pipe[model]    


def compute_acc(y_truth, y_pred):
    """
    

    Parameters
    ----------
    y_truth : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    predict_label = (torch.max(y_pred,1)[1]).data.numpy()        
    target_label = y_truth.data.numpy()
    acc = sum(predict_label == target_label)/y_truth.shape[0]
    return acc


# Define custome autograd function for masked connection.

class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.

        Argumens
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

class Dataset(torch.utils.data.Dataset):
    "dataset iter"
    def __init__(self, data, labels):
        self.data = data
        # make sure that label is in the range [0, n_class-1]
        class_idx = np.unique(labels)
        if (class_idx.min() != 0) | (class_idx.max() != class_idx.shape[0]-1):
            labels_new = np.zeros((labels.shape))
            for loop_idx, idx in enumerate(class_idx):
                labels_new[labels==idx] = loop_idx
        else:
            labels_new = labels
        self.labels = labels_new
        
    def __len__(self):
        "get num of data"
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        ""
        
        # get target data and label
        X = torch.tensor(self.data[index]).type(torch.FloatTensor)
        y = torch.tensor(self.labels[index]).type(torch.LongTensor)
        return X, y
    
    def augment(self, add_size_per_class):
        """
        Data augmention in training set

        Returns
        -------
        None.

        """
        # prepare class pattern
        class_idx = np.unique(self.labels)
        augment_data = np.zeros((class_idx.shape[0]*add_size_per_class, self.data.shape[1]))
        augment_label = np.zeros((class_idx.shape[0]*add_size_per_class))
        # sig_test = np.zeros((class_idx.shape[0], self.data.shape[1]))
        for loop_idx, idx in enumerate(class_idx):
            class_loc = self.labels == idx
            class_data = self.data[class_loc, :]
            augment_data_class = np.zeros((add_size_per_class, self.data.shape[1]))
            for voxel_idx in range(self.data.shape[1]):
                class_in_each_voxel = class_data[:, voxel_idx]
                # significance test
                # _, sig_test[loop_idx, voxel_idx] = stats.normaltest(class_in_each_voxel)
                mean, std = np.mean(class_in_each_voxel), np.std(class_in_each_voxel)
                augment_data_class[:, voxel_idx] = np.random.normal(mean, std, size=add_size_per_class)
            # merge data in augment data and label
            augment_data[loop_idx*add_size_per_class:
                          (loop_idx+1)*add_size_per_class, :] = augment_data_class
            augment_label[loop_idx*add_size_per_class:
                          (loop_idx+1)*add_size_per_class]= np.repeat(idx, add_size_per_class)
            print(f'Finish augmentation in class {int(idx+1)}')
        # merge in data raw
        self.data = np.concatenate((self.data, augment_data), axis=0)
        self.labels = np.concatenate((self.labels, augment_label), axis=0)
        print(f'Datset size after augmentation: {self.data.shape[0]}')
        # return sig_test
        
    def pca_in_ROI(self, ROIs, n_components):
        """
        

        Parameters
        ----------
        ROIs : dict
            Contain roi name and its index
        n_components : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pca_data = np.zeros((self.data.shape[0], len(ROIs.keys())*n_components))
        # start PCA transform
        for idx,roi in enumerate(ROIs.keys()):
            voxel_index = ROIs[roi]
            tmp_data = self.data[:, voxel_index]
            pca = PCA(n_components)
            pca.fit(tmp_data)
            print(np.array(pca.explained_variance_ratio_).sum())
            pca_data[:, idx*n_components:(idx+1)*n_components] = pca.transform(tmp_data)
            print(f'Finish PCA in ROI {roi}')
        self.data = pca_data
        
def train(model, train_set, val_set, batch_size, n_epoch, lr, weight_decay,
          v1=None, v2=None, v4=None, hvc=None, verbose=False):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    train_set : Dataset object
        DESCRIPTION.
    val_set : Dataset object
        DESCRIPTION.
    batch_size : int
        DESCRIPTION.
    n_epoch : int
        DESCRIPTION.
    lr : float
        DESCRIPTION.
    weight_decay : float
        DESCRIPTION.

    Returns
    -------
    None.

    """
    criterion = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p:p.requires_grad,
                                  model.parameters()), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
    train_acc, train_loss, val_acc, val_loss = [], [], [], []
    lambda2, lambda4, lambda_hvc = 0.001, 0.01, 0.1
    # backward pass
    for t in range(n_epoch):
        model.train()
        train_acc_epoch, train_loss_epoch = [], []
        print('======Start Epoch:{:0>2d}/{:0>2d}======'.format(t+1, n_epoch))
        for idx_batch, train_data in enumerate(train_loader, 0):
            X_train, y_train = train_data
            if model.__class__.__name__ == 'HierarchyNet_merge_loss':
                X_input = X_train[:, v1]
                v2_target = X_train[:, v2]
                v4_target = X_train[:, v4]
                hvc_target = X_train[:, hvc]
                # forward
                y_pred, v2_pred, v4_pred, hvc_pred = model(X_input)
                loss = criterion(y_pred, y_train) + lambda2*mse(v2_pred, v2_target) + \
                       lambda4*mse(v4_pred, v4_target) + lambda_hvc*mse(hvc_pred, hvc_target)
                                            
            else:
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)

            # backward and update the params
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # # Check grad
        # with torch.no_grad():
        #     for param in model.parameters():
        #         # mask is also saved in param, but mask.requires_grad=False
        #         # if param.requires_grad: 
        #             # param -= lr * param.grad
        #             # check masked param.grad
        #         if np.array(param.grad).size == 11031*71:
        #             print('masked weight')
        #             print(param.t()[8:20, 34:37])
        #             print('masked grad of weight')
        #             print(param.grad.t()[8:20, 34:37])
                            
            # report acc and loss
            train_acc_epoch.append(compute_acc(y_train, y_pred))
            train_loss_epoch.append(loss.item())
            if verbose:
                print("Idx_train_epoch: {:0>2d}/{:0>2d}.. ".format(t+1, n_epoch),
                      "Idx_train_batch: {:0>4d}/{:0>4d}.. ".format(idx_batch+1, len(train_loader)),
                      "Loss: {:.3f}.. ".format(loss.item()))
        train_acc.append(np.array(train_acc_epoch).mean())
        train_loss.append(np.array(train_loss_epoch).mean())
        print("Training  : Accuracy: {:.3f}; Loss: {:.3f}".format(
            np.array(train_acc_epoch).mean(), np.array(train_loss_epoch).mean()))    
            
        # for validation  
        with torch.no_grad():
            model.eval() 
            val_acc_epoch, val_loss_epoch = [], []
            for idx_batch, val_data in enumerate(val_loader, 0):
                X_val, y_val = val_data
                if model.__class__.__name__ == 'HierarchyNet_merge_loss':
                    X_input = X_train[:, v1]
                    y_pred = model(X_input)[0]
                else:
                    y_pred = model(X_val)
                loss = criterion(y_pred, y_val)
                # report acc and loss
                val_acc_epoch.append(compute_acc(y_val, y_pred))
                val_loss_epoch.append(loss.item())
                if verbose:
                    print("Idx_val_epoch: {:0>2d}/{:0>2d}.. ".format(t+1, n_epoch),
                          "Idx_val_batch: {:0>3d}/{:0>3d}.. ".format(idx_batch+1, len(val_loader)),
                          "Loss: {:.3f}.. ".format(loss.item()))
            val_acc.append(np.array(val_acc_epoch).mean())
            val_loss.append(np.array(val_loss_epoch).mean())
            print("Validation: Accuracy: {:.3f}; Loss: {:.3f}\n".format(
                np.array(val_acc_epoch).mean(), np.array(val_loss_epoch).mean()))
        
    return model.state_dict(), train_acc, train_loss, val_acc, val_loss

def save2cifti(file_path, data, brain_models, map_names=None, volume=None, label_tables=None):
    """
    Save data as a cifti file
    If you just want to simply save pure data without extra information,
    you can just supply the first three parameters.
    NOTE!!!!!!
        The result is a Nifti2Image instead of Cifti2Image, when nibabel-2.2.1 is used.
        Nibabel-2.3.0 can support for Cifti2Image indeed.
        And the header will be regard as Nifti2Header when loading cifti file by nibabel earlier than 2.3.0.
    Parameters:
    ----------
    file_path: str
        the output filename
    data: numpy array
        An array with shape (maps, values), each row is a map.
    brain_models: sequence of Cifti2BrainModel
        Each brain model is a specification of a part of the data.
        We can always get them from another cifti file header.
    map_names: sequence of str
        The sequence's indices correspond to data's row indices and label_tables.
        And its elements are maps' names.
    volume: Cifti2Volume
        The volume contains some information about subcortical voxels,
        such as volume dimensions and transformation matrix.
        If your data doesn't contain any subcortical voxel, set the parameter as None.
    label_tables: sequence of Cifti2LableTable
        Cifti2LableTable is a mapper to map label number to Cifti2Label.
        Cifti2Lable is a specification of the label, including rgba, label name and label number.
        If your data is a label data, it would be useful.
    """
    if file_path.endswith('.dlabel.nii'):
        assert label_tables is not None
        idx_type0 = 'CIFTI_INDEX_TYPE_LABELS'
    elif file_path.endswith('.dscalar.nii'):
        idx_type0 = 'CIFTI_INDEX_TYPE_SCALARS'
    elif file_path.endswith('.dtseries.nii'):
        if len(data.shape) > 1:
            brain_models.header.get_index_map(0).number_of_series_points = data.shape[0]
        else:
            brain_models.header.get_index_map(0).number_of_series_points = 1
            data = data[np.newaxis, :]
        nib.save(nib.Cifti2Image(data.astype(np.float32), brain_models.header), file_path)
        return # jump out of function
    else:
        raise TypeError('Unsupported File Format')

    if map_names is None:
        map_names = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(map_names), "Map_names are mismatched with the data"

    if label_tables is None:
        label_tables = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(label_tables), "Label_tables are mismatched with the data"

    # CIFTI_INDEX_TYPE_SCALARS always corresponds to Cifti2Image.header.get_index_map(0),
    # and this index_map always contains some scalar information, such as named_maps.
    # We can get label_table and map_name and metadata from named_map.
    mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], idx_type0)
    for mn, lbt in zip(map_names, label_tables):
        named_map = cifti2.Cifti2NamedMap(mn, label_table=lbt)
        mat_idx_map0.append(named_map)

    # CIFTI_INDEX_TYPE_BRAIN_MODELS always corresponds to Cifti2Image.header.get_index_map(1),
    # and this index_map always contains some brain_structure information, such as brain_models and volume.
    mat_idx_map1 = cifti2.Cifti2MatrixIndicesMap([1], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
    for bm in brain_models:
        mat_idx_map1.append(bm)
    if volume is not None:
        mat_idx_map1.append(volume)

    matrix = cifti2.Cifti2Matrix()
    matrix.append(mat_idx_map0)
    matrix.append(mat_idx_map1)
    header = cifti2.Cifti2Header(matrix)
    img = cifti2.Cifti2Image(data, header)
    cifti2.save(img, file_path)

def relu(M):
    relu_M = np.zeros_like(M)
    relu_M[M>0] = M[M>0]
    return relu_M

def roi_mask(roi_name, roi_all_names, roi_index):
    """
    Parameters:
    ----------
    roi_name : list or str
    roi_all_names: list
        all of the roi names in roilbl_mmp.csv
    roi_index: ndarray
        the correponding label of each ROI in 32k space
    """
    # start load name
    select_index = []
    if isinstance(roi_name, str):
        roi_tmp_index = roi_all_names.loc[roi_all_names.isin([f'L_{roi_name}_ROI']).any(axis=1)].index[0]+1
        select_index.extend([roi_tmp_index, roi_tmp_index+180])
        mask = np.asarray([True if x in select_index else False for x in roi_index[0]])
    else:
        for name in roi_name:
            roi_tmp_index = roi_all_names.loc[roi_all_names.isin([f'L_{name}_ROI']).any(axis=1)].index[0]+1
            select_index.extend([roi_tmp_index, roi_tmp_index+180])
        mask = np.asarray([True if x in select_index else False for x in roi_index[0]])
    return mask

def get_voxel_location(roi_region, support_path):
    # load roi atlas info
    # make sure that all belowing files are existed
    roi_assign = pd.read_csv(pjoin(support_path, 'HCP-MMP1_visual-cortex1.csv'))
    roi_name_path = pjoin(support_path, 'roilbl_mmp.csv')
    roi_all_names = pd.read_csv(roi_name_path)
    roi_index = sio.loadmat(pjoin(support_path, 'MMP_mpmLR32k.mat'))['glasser_MMP']

    # define rois
    roi_name = roi_assign.loc[roi_assign['ID_in_22Region'].isin(roi_region), 'area_name'].to_list()
    mask = roi_mask(roi_name, roi_all_names, roi_index)

    return mask

def fit_encoding_model(features, brain_resp, visual_mask, feature_type, n_job=4):

    # start computing time
    t1 = time.time()
    # split the dataset into train and test part
    train_features, test_features, brain_train, brain_test = train_test_split( \
        features, brain_resp, test_size=0.2, random_state=42)

    # baseline linear model estimation 
    lr = make_pipeline(StandardScaler(), LinearRegression(n_jobs=n_job))
    lr.fit(train_features, brain_train[:, visual_mask])

    # loop each voxel to compute correlation
    # use inner product to compute correlation
    predictions = lr.predict(test_features)
    std_predictions = (predictions - predictions.mean(axis=0)) / predictions.std(axis=0)
    std_brain_test = (brain_test[:, visual_mask] - brain_test[:, visual_mask].mean(axis=0)) / brain_test[:, visual_mask].std(axis=0)
    # compute correlation
    correlations = np.sum(std_predictions * std_brain_test, axis=0) / (std_predictions.shape[0] - 1)
    lr_r = correlations.mean()
    # report model fitting results
    print(f'{feature_type}, r={lr_r}, tc{(time.time() - t1)}')
    return correlations

def save_encoding_results(results, visual_mask, sub_name, feature_type, out_path):
    # save encoding model map
    temp = nib.load(pjoin('./supportfiles', 'template.dtseries.nii'))
    encoding_map = np.zeros((91282))
    encoding_map[:59412][visual_mask] = results
    # assert whether output path are existed
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    encoding_path = pjoin(out_path, f'{sub_name}_encoding_{feature_type}.dtseries.nii')
    save2cifti(file_path=encoding_path, data=encoding_map, brain_models=temp)

def load_brain_response(sub_name, support_path, normalize_metric='run', clean_code='hp128_s4', runperses=10, trlperrun=100):

    # define beta path
    beta_sub_path = pjoin(support_path, f'{sub_name}_imagenet-beta_{clean_code}_filtered_ridge.npy')

    if not os.path.exists(beta_sub_path):
        raise ValueError('Please run NOD brain response preparation first!')
    else:
        # load beta
        beta_sub = np.load(beta_sub_path)
        # perform data normalization
        if beta_sub.ndim != 2:
            raise AssertionError('check data shape into (n-total-trail,n-brain-size)')
        if normalize_metric == 'run':
            nrun = beta_sub.shape[0] / trlperrun
            for i in range(int(nrun)):
                # run normalization is to demean the run effect 
                beta_sub[i*trlperrun:(i+1)*trlperrun, :] = zscore(beta_sub[i*trlperrun:(i+1)*trlperrun,:], None)
        elif normalize_metric =='session':
            nrun = beta_sub.shape[0] / trlperrun
            nses = nrun/runperses
            for i in range(int(nses)):
                beta_sub[i*trlperrun*runperses:(i+1)*trlperrun*runperses, :] = zscore(beta_sub[i*trlperrun*runperses:(i+1)*trlperrun*runperses, :], None)
        elif normalize_metric=='trial':
            beta_sub = zscore(beta_sub, axis=1)

    return beta_sub


def load_network_features(sub_name, feature_type, support_path):
    # define feature path
    feature_path = pjoin(support_path, 'sub_stim')
    # load features based on different types
    if 'conv' in feature_type or 'fc' in feature_type:
        feature_file_path = pjoin(feature_path, f'{sub_name}_AlexNet.act.h5')
        # load original features
        features_org = Activation()
        features_org.load(feature_file_path)
        # iterate on layers
        # different layer types have different loading metric
        if 'conv' in feature_type:
            features = np.sum(relu(features_org.get(feature_type)), axis=(2,3))
        elif 'fc' in feature_type:
            features = np.squeeze(relu(features_org.get(feature_type)))
    else:
        # get semantic embedding path
        feature_path = pjoin(support_path, 'sub_stim', '%s_%s.npy'%(sub_name, feature_type))
        if not os.path.exists(feature_path):
            raise ValueError('Please generate %s feature first'%feature_type)
        else:
            features = np.load(feature_path)
    return features