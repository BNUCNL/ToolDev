import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import random
from torch.utils.data import Subset
import pandas as pd

# Paths
support_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/Analysis_results/vtc_analysis'
feature_path = pjoin(support_path, 'data', 'feature')
pc_path = pjoin(support_path, 'data', 'pc')
data_path = '/nfs/z1/zhenlab/DNN/ImgDatabase/ImageNet_2012/'
out_path = pjoin(support_path, 'data', 'network')
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Device configuration for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and modify the AlexNet model
def load_modified_alexnet(reduction_weights):
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
            self.fc2_reduction.weight.requires_grad = False  # Freeze PCA weights
            if self.fc2_reduction.bias is not None:
                self.fc2_reduction.bias.requires_grad = False  # Freeze bias if present
            self.fc3 = nn.Linear(60, 1000)
            
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)            
            x = self.fc1(x)            
            x = self.relu(x)
            x = self.relu(self.fc2_reduction(x))          
            x = self.fc3(x)
            return x

    model = ModifiedAlexNet(alexnet, reduction_weights).to(device)
    return model

# Updated data loader function
def get_data_loader(dataset_type, batch_size=128, sample_fraction=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(root=pjoin(data_path, dataset_type), transform=transform)
    
    if sample_fraction is not None:
        class_indices = {class_idx: [] for class_idx in range(len(dataset.classes))}
        for idx, (_, label) in enumerate(dataset.imgs):
            class_indices[label].append(idx)
        
        subset_indices = []
        for class_idx, indices in class_indices.items():
            class_size = len(indices)
            sample_size = int(class_size * sample_fraction)
            sampled_indices = random.sample(indices, sample_size)
            subset_indices.extend(sampled_indices)
            
            print(f"Class {dataset.classes[class_idx]}: Original size = {class_size}, Sampled size = {sample_size}")
        
        dataset = Subset(dataset, subset_indices)
    
    shuffle = True if dataset_type == 'train' else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Training function with progress bar, validation metrics, early stopping, and CSV logging
def train(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001, out_path="./"):
    optimizer = torch.optim.SGD(list(model.fc3.parameters()), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    val_metrics = []
    top5_accuracies = []  # Store Top-5 accuracies to check for early stopping
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        # Training loop with progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                pbar.update(1)
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Calculate validation metrics (Top-1, Top-5 accuracy and loss)
        val_top1, val_top5, val_loss = calculate_topk_accuracy_and_loss(val_loader, model, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Validation Top-1: {val_top1:.2f}%, Top-5: {val_top5:.2f}%, Validation Loss: {val_loss:.4f}')
        
        # Record metrics
        val_metrics.append({"epoch": epoch + 1, "val_loss": val_loss, "top1_accuracy": val_top1, "top5_accuracy": val_top5})
        top5_accuracies.append(val_top5)
        
        # Early stopping check: if the last 3 epochs have less than 1% improvement on average in Top-5 accuracy
        if len(top5_accuracies) >= 4:
            recent_improvements = [top5_accuracies[i] - top5_accuracies[i - 1] for i in range(-3, 0)]
            avg_improvement = sum(recent_improvements) / 3
            if avg_improvement < 1:
                print(f"Early stopping triggered: Average Top-5 accuracy improvement over last 3 epochs is {avg_improvement:.2f}% (below 1%)")
                break
    
    # Save validation metrics to a DataFrame
    val_metrics_df = pd.DataFrame(val_metrics)
    return val_metrics_df

# Top-1 and Top-5 accuracy and loss calculation function
def calculate_topk_accuracy_and_loss(loader, model, criterion):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calculating Validation Metrics", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            total += labels.size(0)
            top1_correct += (top5_pred[:, 0] == labels).sum().item()
            top5_correct += (top5_pred == labels.view(-1, 1)).sum().item()
    
    avg_loss = total_loss / len(loader)
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total
    return top1_accuracy, top5_accuracy, avg_loss

# Main execution
if __name__ == "__main__":
    n_component = 60
    learning_rate = 0.0001
    data_type = 'full'
    reduction_type = 'NMF'

    # load data
    reduction_weight_path = pjoin(pc_path, f'AlexNet_fc1_{reduction_type}_axes.npy')  # Adjust this path to your PCA weight file
    reduction_weights = np.load(reduction_weight_path)[:n_component, :]
    if reduction_type == 'NMF':
        # Apply normalization to NMF weights
        reduction_weights = reduction_weights / np.max(np.abs(reduction_weights))  # Standardize weight range

    # make model
    model = load_modified_alexnet(reduction_weights)
    
    if data_type == 'full':
        n_epoch = 25
        batch_size = 512
        sample_fraction_train = None
        sample_fraction_val = None
    elif data_type == 'simplified':
        n_epoch = 50
        batch_size = 64
        sample_fraction_train = 0.1
        sample_fraction_val = 0.2

    train_loader = get_data_loader('train', batch_size=batch_size, sample_fraction=sample_fraction_train)
    val_loader = get_data_loader('val', batch_size=batch_size, sample_fraction=sample_fraction_val)
    val_metrics_df = train(model, train_loader, val_loader, num_epochs=n_epoch, learning_rate=learning_rate, out_path=out_path)
    # save info
    torch.save(model.state_dict(), pjoin(out_path, f"modified_alexnet_{reduction_type}.pth"))
    val_metrics_df.to_csv(pjoin(out_path, f"train_info_{reduction_type}.csv"), index=False)
    print("Save model weights and training info")
