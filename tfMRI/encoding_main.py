import os
import config
import warnings
from tfMRI_utils import voxel_mask, fit_model, load_NOD_response, load_features

warnings.simplefilter('ignore')
os.chdir(os.path.dirname(__file__))

# parameters setting
sub_names = config.sub_names
support_path = config.support_path
clean_code = config.clean_code
normalize_metric = config.normalize_metric
feature_targets = config.feature_targets
roi_region = config.roi_region

# prepare voxel mask
visual_mask = voxel_mask(roi_region, support_path)

# loop to fit model
for sub_name in sub_names:

    # load_brain_response
    brain_resp = load_NOD_response(sub_name, support_path, normalize_metric=normalize_metric, clean_code=clean_code)
    
    # prepare modeling
    for feature_type in feature_targets:

        # load_features
        features = load_features(sub_name, feature_type, support_path)

        # fit encoding model
        fit_model(features, brain_resp, visual_mask, feature_type, sub_name)

        print('Finish modeling for %s in %s'%(feature_type, sub_name))
