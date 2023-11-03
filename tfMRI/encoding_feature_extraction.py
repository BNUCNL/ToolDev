import os
import math
import torch
import config
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import subprocess
from os.path import join as pjoin
from dnnbrain.dnn.core import Activation
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers import AutoTokenizer, AutoModel


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to compute embeddings in batches
def compute_transformer_embeddings(model, tokenizer, sentences, batch_size=8, progress=False):
    n = len(sentences)
    n_total_batches = math.ceil(n / batch_size)

    all_embeddings = torch.zeros([n, model.config.hidden_size]).to("cuda")
    # assert whether use tqdm to print rate of process
    iterable = range(0, n, batch_size)
    if progress:
        iterable = tqdm(iterable, desc='Computing embeddings', total=n_total_batches)
    # perform iteration
    for i in iterable:
        end_idx = min(i + batch_size, n)
        batch = sentences[i:end_idx]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda")

        with torch.no_grad():
            model_output = model(**encoded_input)
        
        all_embeddings[i:end_idx, :] = mean_pooling(model_output, encoded_input['attention_mask'])
        # print('Compute batch %03d/%03d'%(i+1, n/batch_size))
    return all_embeddings

# Function to compute embeddings in batches
def compute_CLIP_embeddings(model, processor, inputs, output_dim, type, batch_size=8, progress=False):
    n = len(inputs)
    n_total_batches = math.ceil(n / batch_size)
    all_embeddings = torch.zeros([n, output_dim]).to("cuda")
    # assert whether use tqdm to print rate of process
    iterable = range(0, n, batch_size)
    if progress:
        iterable = tqdm(iterable, desc='Computing embeddings', total=n_total_batches)
    # perform iteration
    with torch.no_grad():  # Disable gradient computation
        for i in iterable:
            end_idx = min(i + batch_size, n)
            batch = inputs[i:end_idx]
            if type == 'text':
                encoded_input = processor(text=batch, padding=True, return_tensors='pt').to("cuda")
                features = model.get_text_features(**encoded_input)
            else:
                encoded_input = processor(images=batch, return_tensors="pt").to("cuda")
                features = model.get_image_features(**encoded_input)
            features = features / features.norm(p=2, dim=-1, keepdim=True)  # normalize
            all_embeddings[i:end_idx, :] = features
            # print('Compute batch %03d/%03d'%(i+1, n/batch_size))
    return all_embeddings

def prepare_CLIP_text_feature(sub_name, support_path='./supportfiles'):
    # get semantic embedding path
    feature_path = pjoin(support_path, 'sub_stim')
    text_feature_path = pjoin(feature_path, '%s_CLIP-text.npy'%sub_name)
    if not os.path.exists(text_feature_path):
        print(f'Start preparing CLIP text features for {sub_name}')
        # Load model from HuggingFace Hub
        processor = ChineseCLIPProcessor.from_pretrained(pjoin(model_path, 'chinese-clip-vit-large-patch14'))
        model = ChineseCLIPModel.from_pretrained(pjoin(model_path, 'chinese-clip-vit-large-patch14')).to("cuda")
        # load annotation info
        stim = pd.read_csv(pjoin(support_path, 'nod_stim.csv'), encoding='gbk')
        # obtain annotation info
        label_filtered_file = pjoin(support_path, f'{sub_name}_imagenet-label-filtered.csv')
        df_filtered = pd.read_csv(label_filtered_file)['image_id']
        # Convert the stim DataFrame into a dictionary
        annotation_map = dict(zip(stim['image_name'], stim['annotation']))
        # Use the map function to map image_ids in df_filtered to annotations in stim
        annotations_in_order = df_filtered.map(annotation_map).tolist()
        # obtain semantic embedding
        text_feature = compute_CLIP_embeddings(model, processor, annotations_in_order, model.config.text_config.projection_dim, 
                                        type='text', batch_size=8, progress=True)
        np.save(text_feature_path, text_feature.cpu().numpy())
    else:
        print(f'CLIP text features for {sub_name} have already been generated')

def prepare_CLIP_image_feature(sub_name, support_path='./supportfiles'):
    # get semantic embedding path
    feature_path = pjoin(support_path, 'sub_stim')
    image_feature_path = pjoin(feature_path, '%s_CLIP-image.npy'%sub_name)
    if not os.path.exists(image_feature_path):
        print(f'Start preparing CLIP image features for {sub_name}')
        # Load model from HuggingFace Hub
        stimuli_path = pjoin(model_path, 'stimuli')
        processor = ChineseCLIPProcessor.from_pretrained(pjoin(model_path, 'chinese-clip-vit-large-patch14'))
        model = ChineseCLIPModel.from_pretrained(pjoin(model_path, 'chinese-clip-vit-large-patch14')).to("cuda")
        # load annotation info
        stim = pd.read_csv(pjoin(support_path, 'nod_stim.csv'), encoding='gbk')
        # obtain annotation info
        label_filtered_file = pjoin(support_path, f'{sub_name}_imagenet-label-filtered.csv')
        df_filtered = pd.read_csv(label_filtered_file)
        image_paths = df_filtered['image_name'].to_list()
        image_paths = [pjoin(stimuli_path, x) for x in image_paths]
        # Convert the stim DataFrame into a dictionary
        image_sum = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image_sum.append(image)
        # obtain semantic embedding
        image_feature = compute_CLIP_embeddings(model, processor, image_sum, model.config.vision_config.projection_dim, 
                                        type='image', batch_size=8, progress=True)
        np.save(image_feature_path, image_feature.cpu().numpy())
    else:
        print(f'CLIP image features for {sub_name} have already been generated')

def prepare_transformer_feature(sub_name, support_path='./supportfiles'):
    # get semantic embedding path
    feature_path = pjoin(support_path, 'sub_stim')
    semantic_embedding_path = pjoin(feature_path, '%s_semantic.npy'%sub_name)
    if not os.path.exists(semantic_embedding_path):
        print(f'Start preparing semantic features for {sub_name}')
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(pjoin(model_path, 'paraphrase-multilingual-mpnet-base-v2'))
        model = AutoModel.from_pretrained(pjoin(model_path, 'paraphrase-multilingual-mpnet-base-v2')).to("cuda")
        # load annotation info
        stim = pd.read_csv(pjoin(support_path, 'nod_stim.csv'), encoding='gbk')
        # obtain annotation info
        label_filtered_file = pjoin(support_path, f'{sub_name}_imagenet-label-filtered.csv')
        df_filtered = pd.read_csv(label_filtered_file)['image_id']
        # Convert the stim DataFrame into a dictionary
        annotation_map = dict(zip(stim['image_name'], stim['annotation']))
        # Use the map function to map image_ids in df_filtered to annotations in stim
        annotations_in_order = df_filtered.map(annotation_map).tolist()
        # obtain semantic embedding
        semantic_embedding = compute_transformer_embeddings(model, tokenizer, annotations_in_order, batch_size=8, progress=True)
        np.save(semantic_embedding_path, semantic_embedding.cpu().numpy())
    else:
        print(f'Semantic features for {sub_name} have already been generated')

def prepare_AlexNet_feature(sub_name):
    # get AlexNet embedding path
    feature_path = pjoin(support_path, 'sub_stim')
    feature_file_path = pjoin(feature_path, f'{sub_name}_AlexNet.act.h5')
    stimcsv = pjoin(feature_path, f'{sub_name}_imagenet.stim.csv')
    if not os.path.exists(feature_file_path):
        # make sure dnnbrain have been installed
        print('activation file preparing')
        command = f"dnn_act -net AlexNet -layer conv1 conv2 conv3 conv4 conv5 fc1 fc2 fc3 -stim ./{stimcsv} -out ./{feature_file_path}"
        verbose = 0
        ret = subprocess.run(command,shell=True,
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                encoding="utf-8",timeout=None)
        if ret.returncode == 0 and verbose:
            print("success:",ret)
        else:
            print("error:",ret)

# change current path
os.chdir(os.path.dirname(__file__))

# Set up the argument parser
parser = argparse.ArgumentParser(description='Model fitting for fMRI data.')
parser.add_argument('--sub_name', type=str, nargs='+', default=['sub-04'], 
                    help='Subject names', required=True)

# Parse the arguments
args = parser.parse_args()

# parameters setting
support_path = config.support_path
dataset_path = config.dataset_path
model_path = config.model_path

# parameters setting for user input
sub_names = args.sub_name

# loop to generate features
for sub_name in sub_names:
    prepare_CLIP_text_feature(sub_name)
    prepare_CLIP_image_feature(sub_name)
    prepare_transformer_feature(sub_name)
    prepare_AlexNet_feature(sub_name)
