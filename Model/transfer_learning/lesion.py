import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as pjoin

# Paths and device configuration
data_type = 'full'
support_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/Analysis_results/vtc_analysis'
feature_path = pjoin(support_path, 'data', 'feature')
pc_path = pjoin(support_path, 'data', 'pc')
out_path = pjoin(support_path, 'data', 'network')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '/nfs/z1/zhenlab/DNN/ImgDatabase/ImageNet_2012/'


# Load the modified AlexNet model with all trained weights
def load_modified_alexnet(reduction_weights, model_weight_path):
    alexnet = models.alexnet(pretrained=True)
    for param in alexnet.parameters():
        param.requires_grad = False

    class ModifiedAlexNet(nn.Module):
        def __init__(self, original_model, reduction_weights):
            super(ModifiedAlexNet, self).__init__()
            self.features = original_model.features
            self.avgpool = original_model.avgpool
            self.fc1 = original_model.classifier[1]
            self.relu = nn.ReLU(inplace=True)
            self.fc2_reduction = nn.Linear(4096, 60)
            self.fc2_reduction.weight.data = torch.tensor(reduction_weights, dtype=torch.float32)
            self.fc2_reduction.weight.requires_grad = False
            if self.fc2_reduction.bias is not None:
                self.fc2_reduction.bias.requires_grad = False
            self.fc3 = nn.Linear(60, 1000)
            
        def forward(self, x, lesion_neurons=None):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)            
            x = self.fc1(x)            
            x = self.relu(x)
            x = self.fc2_reduction(x)
            if lesion_neurons is not None:
                x[:, lesion_neurons] = 0  # Set the output of specified neurons to zero
            x = self.relu(x)
            x = self.fc3(x)
            return x

    model = ModifiedAlexNet(alexnet, reduction_weights).to(device)
    
    # Load the full model weights
    state_dict = torch.load(model_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    
    return model

# Data loader function
def get_val_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'), transform=transform)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False), val_dataset.classes


# Per-class and overall Top-5 accuracy calculation function
def calculate_per_class_top5_accuracy(loader, model, lesion_neurons=None, num_classes=1000):
    model.eval()
    # Initialize per-class counts and overall counts
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    overall_correct = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, lesion_neurons=lesion_neurons)
            
            # Get top 5 predictions for each sample
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            
            # Update per-class counts and overall correct count
            for label, top5 in zip(labels, top5_pred):
                class_total[label] += 1
                if label in top5:
                    class_correct[label] += 1
                    overall_correct += 1
    
    # Calculate per-class Top-5 accuracy
    class_top5_accuracy = [(100 * class_correct[i] / class_total[i]) if class_total[i] > 0 else 0 for i in range(num_classes)]
    
    # Calculate overall Top-5 accuracy using class_total for overall total
    overall_total = sum(class_total)
    overall_top5_accuracy = 100 * overall_correct / overall_total if overall_total > 0 else 0
    print(f"Overall Top-5 Accuracy: {overall_top5_accuracy:.2f}%")
    
    return class_top5_accuracy


# Main experiment
if __name__ == "__main__":
    # Load PCA weights and initialize model with full weights
    reduction_type = 'NMF'
    n_component = 60
    model_weight_path = pjoin(out_path, f"modified_alexnet_{reduction_type}.pth")  # Path to the previously saved model weights
    output_csv = pjoin(out_path, 'nmf_neuron_lesion_results.csv')

    # load data
    reduction_weight_path = pjoin(pc_path, f'AlexNet_fc1_{reduction_type}_axes.npy')  # Adjust this path to your PCA weight file
    reduction_weights = np.load(reduction_weight_path)[:n_component, :]
    if reduction_type == 'NMF':
        # Apply normalization to NMF weights
        reduction_weights = reduction_weights / np.max(np.abs(reduction_weights))  # Standardize weight range

    # load model
    model = load_modified_alexnet(reduction_weights, model_weight_path)
    
    # Load validation data
    val_loader, class_names = get_val_loader(batch_size=512)
    
    # Initialize results dictionary
    results = {}

    # Record per-class top-5 accuracy of the original model
    print("Calculating original model Top-5 accuracy...")
    original_class_top5_accuracy = calculate_per_class_top5_accuracy(val_loader, model)
    results["None (original)"] = original_class_top5_accuracy
    
    # Perform lesion experiments on each neuron in fc2_reduction
    for neuron_idx in range(60):
        print(f"Lesioning neuron {neuron_idx} in fc2_reduction")
        class_top5_accuracy = calculate_per_class_top5_accuracy(val_loader, model, lesion_neurons=[neuron_idx])
        results[f"Neuron {neuron_idx}"] = class_top5_accuracy
    
    # Convert results dictionary to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.index = class_names  # Set index to class names for better readability
    results_df.to_csv(output_csv)
    print(f"Results saved to {output_csv}")
