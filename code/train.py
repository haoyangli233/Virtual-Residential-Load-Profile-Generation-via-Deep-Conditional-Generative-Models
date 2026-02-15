from tqdm import tqdm
import torch
import utils
import time
import logging
import itertools
import os
import json

def train_medoid_vae(model, num_epochs, dataloader, optimizer, device, model_name):
    '''
    Function to train the medoid VAE model

    The medoid VAVE takes in the weather, cluster, time, and real energy data as input,
    Outputs the synthetic energy data for the medoid user of the cluster.

    Args:
    model (torch.nn.Module): The model to be trained
    num_epochs (int): Number of epochs to train the model
    dataloader (torch.utils.data.DataLoader): DataLoader object
    optimizer (torch.optim.Optimizer): Optimizer object
    device (torch.device): Device to train the model on
    model_name (str): Name of the model to be saved
    '''

    best_loss = float('inf')
    model.train()
    pbar = tqdm(range(num_epochs), desc="Training Progress", ncols = 100)

    patience = 0

    for epoch in pbar:
        total_loss = 0.0
        num_batches = 0

        for _, (weather, cluster, time, _, _, real_energy) in enumerate(dataloader):
            weather = weather.to(device)
            cluster = cluster.to(device).int()
            time = time.to(device)
            real_energy = real_energy.to(device)

            optimizer.zero_grad()
            output, mu, logvar = model(weather, cluster, time, real_energy)
            loss = utils.vae_loss_function(output, real_energy, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches
        pbar.set_postfix({'current loss': epoch_loss})

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1

        if patience == 5:
            print('Early stopping')
            break

    print(f'Finished training, best loss: {best_loss:.4f}')
    return best_loss, best_model

def train_m2s_vae(model_m2s, model_medoid, num_epochs, loader, optimizer, device, model_name):
    '''
    Function to train the medoid to synthetic VAE model
    
    The medoid to synthetic VAE takes in the synthetic energy data for the medoid user of the cluster,
    weather, cluster, time, statistical, spike type, and spike magnitude data as input,
    Outputs the synthetic energy data for the synthetic user of the cluster.
    
    Args:
    model_m2s (torch.nn.Module): The model to be trained
    model_medoid (torch.nn.Module): The medoid model
    num_epochs (int): Number of epochs to train the model
    loader (torch.utils.data.DataLoader): DataLoader object
    optimizer (torch.optim.Optimizer): Optimizer object
    device (torch.device): Device to train the model on
    model_name (str): Name of the model to be saved
    '''
    
    best_loss = float('inf')
    
    model_m2s.train()
    model_medoid.eval()

    pbar = tqdm(range(num_epochs), desc="Training Progress", ncols = 100)

    patience = 0

    for epoch in pbar:
        total_loss = 0.0
        num_batches = 0

        for idx, (weather, cluster, time, statistical, spike, real_energy) in enumerate(loader):
            weather = weather.to(device)
            cluster = cluster.to(device).int()
            time = time.to(device)
            statistical = statistical.to(device)
            spike_type = spike[:, :, 0:1].to(dtype=torch.int32)
            spike_magnitude = spike[:, :, 1:].to(dtype=torch.float32)
            spike_type = spike_type.to(device)
            spike_magnitude = spike_magnitude.to(device)
            real_energy = real_energy.to(device)

            with torch.no_grad():
                noise = torch.randn(real_energy.shape[0], real_energy.shape[1], utils.get_m_latent_dim()).to(device)
                medoid_synthetic = model_medoid.lstm_decoder(noise, weather, cluster, time)

            optimizer.zero_grad()
            synthetic_profile, mu, logvar = model_m2s(medoid_synthetic, weather, cluster, time, statistical, spike_type, spike_magnitude)
            loss = utils.vae_loss_function(synthetic_profile, real_energy, mu, logvar)

            loss += 0.5 * utils.gradient_loss(synthetic_profile, statistical)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        epoch_loss = total_loss / num_batches
        pbar.set_postfix({'current loss': epoch_loss})

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model_m2s.state_dict()
            patience = 0
        else:
            patience += 1

        if patience == 5:
            print('Early stopping')
            break
    
    print(f'Finished training, best loss: {best_loss:.4f}')
    return best_loss, best_model