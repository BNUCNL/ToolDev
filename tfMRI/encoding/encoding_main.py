import os
import config
import argparse
import warnings
from tfMRI_utils import get_voxel_location, fit_encoding_model, save_encoding_results, load_brain_response, load_network_features

warnings.simplefilter('ignore')
os.chdir(os.path.dirname(__file__))

# Set up the argument parser
parser = argparse.ArgumentParser(description='Model fitting for fMRI data.')
parser.add_argument('--sub_name', type=str, nargs='+', default=['sub-04'], 
                    help='Subject names', required=True)
parser.add_argument('--roi_region', type=int, nargs='+', default=[1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18],
                    help='List of ROI regions', required=True)
parser.add_argument('--feature_targets', type=str, nargs='+', default=['CLIP-image', 'BERT'], 
                    choices=['CLIP-image', 'CLIP-text', 'BERT', 'ViT'], help='Types of network features', required=True)

# Parse the arguments
args = parser.parse_args()

# parameters setting for config
out_path = config.out_path
support_path = config.support_path
clean_code = config.clean_code
normalize_metric = config.normalize_metric

# parameters setting for user input
sub_names = args.sub_name
roi_region = args.roi_region
feature_targets = args.feature_targets

# prepare voxel mask
visual_mask = get_voxel_location(roi_region, support_path)

# loop to fit model
for sub_name in sub_names:

    # load_brain_response
    brain_resp = load_brain_response(sub_name, support_path, normalize_metric=normalize_metric, clean_code=clean_code)
    
    # prepare modeling
    for feature_type in feature_targets:

        # load_features
        features = load_network_features(sub_name, feature_type, support_path)

        # fit encoding model
        results = fit_encoding_model(features, brain_resp, visual_mask, feature_type)

        # save encodeing results
        save_encoding_results(results, visual_mask, sub_name, feature_type, out_path)

        print('Finish modeling for %s in %s'%(feature_type, sub_name))
