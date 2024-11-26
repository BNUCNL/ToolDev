import os,re 
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join as pjoin
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib import font_manager
from scipy.stats import zscore
import nimfa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import csv, h5py
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize

# Function to resize image if needed
def resize_image(path, size=(100,100)):
    image = Image.open(path).convert('RGB')
    resized_image = image.resize(size, Image.LANCZOS)
    return np.array(resized_image)

def visualize_components(target_matrix, stimulus, result_path, component_prefix, n_components, n_show=5, grid_size=(1, 5), fig_size=(15, 5), index_type='positive'):
    """
    Visualize top components from the target matrix and save the images.

    Parameters:
    - target_matrix: NMF or PCA components matrix.
    - stimulus: List or array containing the stimulus images.
    - result_path: Path to save the results.
    - component_prefix: Prefix for the saved images (e.g., 'dataset_model_layer').
    - n_components: Number of components to visualize.
    - n_show: Number of top images to display for each component.
    - grid_size: Size of the grid for displaying images (default is 1 row, 5 columns).
    - fig_size: Size of the figure (default is (15, 5)).
    - index_type: 'positive' for top positive components, 'negative' for top negative components.
    """
    # Create the result path if it doesn't exist
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Loop over components
    for component_idx in range(n_components):
        # Get vector for the current component
        axes_vector = target_matrix[:, component_idx]
        
        # Determine top indices based on index_type
        if index_type == 'positive':
            top_indices = np.argsort(axes_vector)[-n_show:][::-1].astype(int)
        elif index_type == 'negative':
            top_indices = np.argsort(axes_vector)[:n_show].astype(int)
        else:
            raise ValueError("index_type must be 'positive' or 'negative'")
        
        # Extract the images for the selected indices
        pole_images = [resize_image(stimulus[idx]) for idx in top_indices]

        # Create figure and axes
        fig, axes = plt.subplots(*grid_size, figsize=fig_size, gridspec_kw={'wspace': 0, 'hspace': 0})

        # Plot images
        for ax_idx, img in enumerate(pole_images):
            row = ax_idx // grid_size[1]
            col = ax_idx % grid_size[1]
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')

        # Save the figure
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join(result_path, f'{component_prefix}-{component_idx+1:02d}-{index_type}.jpg'), dpi=300)
        plt.close()
        print(f'Finished plotting image for {component_idx+1}')


def projection_on_nmf(net_feature):
    """
    This function loads natural color features, applies ReLU activation, scales the data,
    and transforms the features into NMF space using a pre-trained model.

    Parameters:
    target_type (str): The target type to load features for (e.g., 'pure_background').
    feature_path (str): The path where the feature .npy files are stored.
    pc_path (str): The path where the pre-trained NMF model is stored.

    Returns:
    np.ndarray: The transformed NMF features (color_natural_score).
    """
    
    # Apply ReLU (rectified linear unit) to remove negative values
    net_feature = np.maximum(0, net_feature)

    # Scale the feature values to be between 0 and 1
    scaler = MinMaxScaler()
    net_feature = scaler.fit_transform(net_feature)

    # Load the pre-trained NMF model and transform the features into NMF space
    nmf = joblib.load(pjoin(pc_path, f'AlexNet_fc1_nmf_model.pkl'))
    net_dim_score = nmf.transform(net_feature)
    return net_dim_score

# Function to load image paths from 'things' or 'nod' datasets
def load_image_paths(dataset_name):
    # define stim csv
    stim_csv = pjoin(stim_path, f'{dataset_name}.stim.csv')
    image_paths = []
    with open(stim_csv, 'r') as labels:
        reader = csv.reader(labels)
        for i, row in enumerate(reader):
            if i == 1:
                stimuli_path = row[0].split('=')[-1]
            if i > 3:
                image_paths.append(pjoin(stimuli_path, row[0]))
    print(f"Load {dataset_name} stimuli : {len(image_paths)}")
    return np.array(image_paths)


# Function to handle 'sum' case by combining stimuli from 'things', 'nod', and 'nsd'
def load_sum_stimuli():
    # Load NSD stimuli
    nsd_stimulus = load_image_paths('nsd')
    
    # Load 'things' stimuli
    things_stimulus = load_image_paths('things')

    # Load 'nod' stimuli
    nod_stimulus = load_image_paths('nod')

    # Concatenate NSD, things, and nod stimuli into a single ndarray
    stimulus = np.concatenate((nsd_stimulus, things_stimulus, nod_stimulus), axis=0)

    print(f"Total stimuli shape: {stimulus.shape}")  # Check the final shape of the concatenated stimuli
    return stimulus

# define info
image_size = (425, 425)  # Resize images to 64x64 for display

# set font
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams.update({'font.size': 9, 'mathtext.fontset': 'stix'})
arial_font_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/utils/font/arial.ttf'
font_manager.fontManager.addfont(arial_font_path)
mpl.rcParams['font.family'] = 'Arial'

# define path 
support_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/Analysis_results/vtc_analysis'
data_path = pjoin(support_path, 'data')
feature_path = pjoin(support_path, 'data', 'feature')
pc_path = pjoin(support_path, 'data', 'pc')
stim_path = pjoin(support_path, 'stim')
util_path = pjoin(support_path, 'utils')
model_path = pjoin(support_path, 'data', 'model')
encoding_result_path = pjoin(support_path, 'result', 'encoding')
beta_path = pjoin(support_path, 'data', 'beta')
result_path = pjoin(support_path, 'result')
weight_path = pjoin(support_path, 'data', 'weight')
floc_path = pjoin(data_path, 'floc')

# Example usage of the function
dataset_name = 'nsd'
model_name = 'AlexNet'
layer_name = 'fc1'
method = 'bnmf'
index_type='positive'
n_components = 60
n_show = 100
grid_size = (10, 10)
fig_size = (20, 20)  # Each subplot will be 2x2
inter_path = pjoin(result_path, 'interpretation', 'nod_component', 'test')
if not os.path.exists(inter_path):
    os.makedirs(inter_path)
target_matrix = np.load(pjoin(pc_path, f'{dataset_name}_{method}_transformed_{model_name}_{layer_name}.npy'))

# load stimulus
stimulus = load_image_paths(dataset_name)

# load feature 
dataset_feature = np.load(pjoin(feature_path, f'{dataset_name}_{model_name}_{layer_name}.npy'))
dataset_dim_score = projection_on_nmf(dataset_feature)
np.save(pjoin(pc_path, f'{dataset_name}_nmf_transformed_{model_name}_{layer_name}.npy'), dataset_dim_score)

visualize_components(
    target_matrix=dataset_dim_score,  # Pre-loaded target matrix
    stimulus=stimulus,            # List/array of stimulus images
    result_path=inter_path,
    component_prefix=f'{dataset_name}_{model_name}_{layer_name}_{method}',  # Prefix for saved files
    n_components=n_components,              # Example: visualize 10 components
    n_show=n_show,                     # Show top 5 images per component
    grid_size=grid_size,             # 1 row and 5 columns
    fig_size=fig_size,             # Adjust figure size as needed
    index_type=index_type
)
