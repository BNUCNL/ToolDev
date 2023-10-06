import os
import numpy as np
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio 
import nibabel as nib
from nibabel import cifti2
import scipy.io as sio

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