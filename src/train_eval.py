import os
import random
import time
import numpy as np
from itertools import cycle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from src.Datasets import *
from src.load_dataset import load_dataset
from src.model import *

def train(model_name: str, model: nn.Module, device: torch.device, data_loader: DataLoader, optimizer: torch.optim.Optimizer, pbar: tqdm) -> float:
    """Trains a GNN model.

    Args
    - model: nn.Module, GNN model, already placed on device
    - device: torch.device
    - data_loader: pyg.loader.DataLoader
    - optimizer: torch.optim.Optimizer
    - loss_fn: nn.Module

    Returns: loss
    - loss: float, avg loss across epoch
    """
    model.train()
    total_recon_loss = 0
    total_kl_div = 0

    pbar.reset(len(data_loader))
    pbar.set_description('Training')

    for step, batch in enumerate(data_loader):
        batch = batch.to(device)
        batch_size = batch.batch.max().item()

        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        #neg_edge_index = batch.neg_edge_index.to(device)
        
        z = model.encode(data = batch)

        recon_loss = model.recon_loss(z, edge_index)
        total_recon_loss += recon_loss.item() * batch_size
        
        if model_name == 'VGAE':
            kl_div = model.kl_loss() #don't need to specify mu or loss since it will use the last one
            total_kl_div += kl_div.item() * batch_size
        else:
            kl_div = 0

        loss = recon_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update()

    total_loss = total_recon_loss + total_kl_div    
    avg_recon_loss = total_recon_loss / len(data_loader)
    avg_kl_div = total_kl_div / len(data_loader)
    return avg_recon_loss, avg_kl_div

def eval(model: nn.Module, device: torch.device, loader: DataLoader,
         pbar: tqdm) -> np.array:
    """Evaluates the model by extracting the embeddings.

    Args
    - model: nn.Module, GNN model, already moved to device
    - device: torch.device
    - loader: DataLoader
    - pbar: tqdm, progress bar

    Returns: np.array 
    - extracting embeddings (before the decoding layer)
    """
    model.eval()
    save_embeddings = np.array([])

    pbar.reset(total=len(loader))
    pbar.set_description('Evaluating')
    
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            #print(model.encode(batch, extract=True).shape)
            embedding = model.encode(extract=True, data=batch)
            embedding = embedding.cpu()
            embedding = embedding.reshape((56, -1, 32))

            a = embedding[38:41, :, :]
            print(a.shape)
            b = embedding[53, :, :]
            print(b.shape)
            embedding = np.concatenate((a, b), axis=1)
            print(embedding.shape)

        #need to figure out how to get the right features from the graph object
        if save_embeddings.shape[0] == 0:
            save_embeddings = embedding
        else:
            save_embeddings = np.concatenate([save_embeddings, embedding], axis=0)
        pbar.update()

    return save_embeddings

def start_training(save_path, data_config, model_config, train_config, device):

    # Sample and fix a random seed if not set in train_config
    if 'seed' not in train_config:
        train_config['seed'] = random.randint(0, 9999)
    seed = train_config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Get model class
    model_class = get_model_class(model_config['name'])
    
    # Initialize dataset
    dataset, model_config = load_dataset(data_config, model_config)
    model = model_class(GraphEncoder(model_config = model_config), model_config=model_config).to(device)
    
    #Initialize dataloaders
    train_loader = DataLoader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=True)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = train_config['learning_rate'])

    # Start training
    pbar = tqdm()
    for epoch in range(1, 1 + train_config['num_epochs']):
        recon_loss, kl_div = train(model_config['name'], model, device, train_loader, optimizer, pbar)
        loss = recon_loss + kl_div
        
        #train_result = eval(model, device, train_loader, evaluator, pbar)
        #val_result = eval(model, device, valid_loader, evaluator, pbar)

        tqdm.write(f'Epoch {epoch:02d}, recon_loss: {recon_loss:.4f}, kl_div: {kl_div:.4f}')
            
        #update the best model after each epoch
        if epoch == 1 or loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path + '/best.pth')
            print('Best model saved') 

def extract_features(save_path, data_config, model_config, train_config, device):

    # Get model class and load data
    model_class = get_model_class(model_config['name'])
    dataset, model_config = load_dataset(data_config, model_config, extract=True)

    #Use Pytorch's built in GAE/VGAE
    model = model_class(GraphEncoder(model_config = model_config), model_config=model_config).to(device)
    
    #Initialize dataloaders
    loader = DataLoader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=False)

    #Get best model
    model.load_state_dict(torch.load(save_path + '/best.pth'))
    pbar = tqdm()
    embeddings = eval(model, device, loader, pbar)

    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
    print("Saved Features: " + os.path.join(save_path, 'embeddings.npy'))

