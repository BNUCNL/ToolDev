import os
import torch
import config
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from os.path import join as pjoin
from sklearn.model_selection import train_test_split

class Encoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[128, 256], latent_dim=16):
        super(Encoder, self).__init__()

        # Semantic encoder
        self.fc1_semantic = nn.Linear(input_dim, hidden_dims[0])
        self.fc2_semantic = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.z_mean_semantic = nn.Linear(hidden_dims[1], latent_dim)
        self.z_log_var_semantic = nn.Linear(hidden_dims[1], latent_dim)

        # Visual encoder
        self.fc1_visual = nn.Linear(input_dim, hidden_dims[0])
        self.fc2_visual = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.z_mean_visual = nn.Linear(hidden_dims[1], latent_dim)
        self.z_log_var_visual = nn.Linear(hidden_dims[1], latent_dim)

        # Text encoder
        self.fc1_text = nn.Linear(input_dim, hidden_dims[0])
        self.fc2_text = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.z_mean_text = nn.Linear(hidden_dims[1], latent_dim)
        self.z_log_var_text = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x, data_type="ViT"):

        # for bert data
        if data_type == "BERT":
            # text encoder forward
            h_text = F.relu(self.fc1_text(x))
            h_text = F.relu(self.fc2_text(h_text))
            z_mean_text = self.z_mean_text(h_text)
            z_log_var_text = self.z_log_var_text(h_text)
            # semantic encoder forward
            h_semantic = F.relu(self.fc1_semantic(x))
            h_semantic = F.relu(self.fc2_semantic(h_semantic))
            z_mean_semantic = self.z_mean_semantic(h_semantic)
            z_log_var_semantic = self.z_log_var_semantic(h_semantic)
            return z_mean_text, z_mean_semantic, z_log_var_text, z_log_var_semantic

        # for vit data
        elif data_type == "ViT":
            # visual encoder forward
            h_visual = F.relu(self.fc1_visual(x))
            h_visual = F.relu(self.fc2_visual(h_visual))
            z_mean_visual = self.z_mean_visual(h_visual)
            z_log_var_visual = self.z_log_var_visual(h_visual)
            # semantic encoder forward
            h_semantic = F.relu(self.fc1_semantic(x))
            h_semantic = F.relu(self.fc2_semantic(h_semantic))
            z_mean_semantic = self.z_mean_semantic(h_semantic)
            z_log_var_semantic = self.z_log_var_semantic(h_semantic)
            return z_mean_visual, z_mean_semantic, z_log_var_visual, z_log_var_semantic


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256, output_dim=768):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(h))
        return x_recon

class CVAE(nn.Module):
    def __init__(self, encoder, decoder, data_type):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.data_type = data_type

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # get encoder output
        z_mean_specific, z_mean_shared, z_log_var_specific, z_log_var_shared = self.encoder(x, self.data_type)
        # merge z mean and log var
        z_means = torch.cat((z_mean_specific.unsqueeze(0), z_mean_shared.unsqueeze(0)), dim=0)
        z_log_vars = torch.cat((z_log_var_specific.unsqueeze(0), z_log_var_shared.unsqueeze(0)), dim=0)
        # reparameterization 
        z_specific = self.reparameterize(z_mean_specific, z_log_var_specific)
        z_shared = self.reparameterize(z_mean_shared, z_log_var_shared)
        z = torch.cat((z_specific, z_shared), dim=1)
        # get decoder output
        x_recon = self.decoder(z)
        return x_recon, z_means, z_log_vars

# Loss functions
def reconstruction_loss(recon_x, x):
    return F.mse_loss(recon_x, x)

def kl_divergence_loss(mus, log_vars):
    loss_specific = -0.5 * torch.sum(1 + log_vars[0] - mus[0].pow(2) - log_vars[0].exp())
    loss_shared = -0.5 * torch.sum(1 + log_vars[1] - mus[1].pow(2) - log_vars[1].exp())
    return loss_specific + loss_shared

class VectorDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# define parameters
os.chdir(os.path.dirname(__file__))
support_path = config.support_path
data_path = pjoin(support_path, 'sub_stim')
data_type = 'ViT'
batch_size = 32
device = "cuda"

for data_type in ['ViT', 'BERT']:
    # load data
    data_file = np.load(pjoin(data_path, f'sub-04_{data_type}.npy'))
    data_matrix = torch.from_numpy(data_file)
    train_data, test_data = train_test_split(data_matrix, test_size=0.2, random_state=42)

    # intialize network
    encoder = Encoder()
    decoder = Decoder()
    cvae = CVAE(encoder, decoder, data_type)
    cvae = cvae.to(device)

    # creating PyTorch Datasets
    train_dataset = VectorDataset(train_data)
    test_dataset = VectorDataset(test_data)
    # Creating DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # start training the model
    optimizer = optim.Adam(cvae.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        for i, train_batch in enumerate(train_loader):
            train_batch = train_batch.to(device)
            optimizer.zero_grad()
            recon_batch, mus, log_vars = cvae(train_batch)
            loss = reconstruction_loss(recon_batch, train_batch) + kl_divergence_loss(mus, log_vars)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"{data_type}: Train Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader.dataset)}")

    # test on the model
    cvae.eval()
    with torch.no_grad():
        test_loss = 0
        for i, test_batch in enumerate(test_loader):
            test_batch = test_batch.to(device)
            recon_batch, _, _ = cvae(test_batch)
            # compute loss
            mse_loss = F.mse_loss(recon_batch, test_batch)
            test_loss += mse_loss.item()

        print(f"{data_type}: Test Loss: {test_loss/len(test_loader.dataset)}")
