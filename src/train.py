import os
import random
import time
import numpy as np
from itertools import cycle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.Datasets import MSADataset
from src.load_dataset import load_dataset
from src.model import *

def train(model: nn.Module, device: torch.device, data_loader: DataLoader, optimizer: torch.optim.Optimizer, pbar: tqdm) -> float:
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
    total_loss = 0

    pbar.reset(len(data_loader))
    pbar.set_description('Training')

    for step, batch in enumerate(data_loader):
        batch = batch.to(device)
        batch_size = batch.batch.max().item()

        x = batch.x.to(device)
        pos_edge_index = batch.edge_index.to(device)
        neg_edge_index = batch.neg_edge_index.to(device)
        
        z = model.encode(x, pos_edge_index)
        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        total_loss += loss.item() * batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update()
        
    avg_loss = total_loss / len(data_loader)
    return avg_loss

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
    dataset = load_dataset(data_config, model_config)
    model = model_class(GraphEncoder(model_config = model_config)).to(device)
    
    #Initialize dataloaders
    train_loader = DataLoader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=True)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Start training
    pbar = tqdm()
    for epoch in range(1, 1 + train_config['num_epochs']):
        loss = train(model_class, device, train_loader, optimizer, pbar)

        #train_result = eval(model, device, train_loader, evaluator, pbar)
        #val_result = eval(model, device, valid_loader, evaluator, pbar)

        tqdm.write(f'Epoch {epoch:02d}, loss: {loss:.4f}')
            
        #update the best model after each epoch
        if epoch == 0 or loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path + '/best.pth')
            print('Best model saved') 

