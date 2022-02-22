import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.Datasets import MSADataset

from util.load_dataset import load_dataset
from util.model import *

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
    if model_config['name'] == 'ProtTP':
        dataset = load_dataset(data_config, model_config)
        model = model_class(model_config, dataset).to(device)
        # Initialize dataloaders
        data_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True)
    elif model_config['name'] == 'MSATP':
        dataset, MSAdataset = load_dataset(data_config, model_config)
        model = model_class(model_config, dataset, MSAdataset).to(device)
        # Initialize dataloaders
        data_loader = DataLoader(MSAdataset, batch_size=train_config['batch_size'], shuffle=True)
   
    # Initialize optimizer
    model.init_optimizer(train_config)

    # Start training
    best_loss = 0
    for epoch in range(train_config['num_epochs']):
        for i, (x, a) in enumerate(data_loader):
            # Forward pass
            x = x.to(device)#.view(-1, dataset.input_dim)
            a = a.to(device)#.view(-1, input_dim)

            losses = model(x, a)
            model.optimize()
            
        print ("Epoch[{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Attribute Loss: {:.4f}" 
            .format(epoch+1, train_config['num_epochs'], losses['reconst_loss'], losses['kl_div'], losses['attr_loss']))
        
        #update the best model after each epoch
        if epoch == 0 or model.total_loss < best_loss:
            best_loss = model.total_loss
            torch.save(model.state_dict(), save_path + '/best.pth')
            print('Best model saved')   

