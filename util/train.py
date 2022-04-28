import os
import random
import time
import numpy as np
from itertools import cycle
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
        model = model_class(model_config = model_config, dataset = dataset).to(device)
        # Initialize dataloaders
        data_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True)
    elif model_config['name'] == 'MSATP':
        #the first is the labeled dataset
        #the second is the dataset of MSAs
        dataset, MSAdataset = load_dataset(data_config, model_config)
        model = model_class(model_config=model_config, dataset=dataset, MSAdataset=MSAdataset).to(device)
        # Initialize dataloaders
        # Drop the last batch so the sizes match up
        data_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, drop_last=True)
        MSAdata_loader = DataLoader(MSAdataset, batch_size=train_config['batch_size'], shuffle=True, drop_last=True)
   
    # Initialize optimizer
    model.init_optimizer(train_config)

    # Start training
    best_loss = 0
    start_time = time.time()
    for epoch in range(train_config['num_epochs']):
        epoch_start_time = time.time()
        
        #set all the cuumlative losses to zero
        cum_losses = model.losses
        for key in cum_losses: cum_losses[key] = 0

        if model_config['name'] == 'ProtTP':
            for i, (x, a) in enumerate(data_loader):
                # Forward pass
                x = x.to(device)#.view(-1, dataset.input_dim)
                a = a.to(device)#.view(-1, input_dim)
                model(x, a)
                model.optimize()
                for key in model.losses: cum_losses[key] += model.losses[key]
        
        if model_config['name'] == 'MSATP':
            # each epoch refers to the larger dataset
            # loop smaller dataset mutliple times in each epoch
            if not model.attributes:
                loaders = zip(MSAdata_loader, data_loader)
            elif len(data_loader) < len(MSAdata_loader):
                loaders = zip(MSAdata_loader, cycle(data_loader))
            else:
                loaders = zip(cycle(MSAdata_loader), data_loader)
            
            for i, (x1, (x2, a)) in enumerate(loaders):
                x1 = x1.to(device)
                a = a.to(device)
                x2 = x2.to(device)

                #alternate forward passes between the two tracks during training
                model(x1, a, track=1)
                #second track will only forwad pass if the attribute decoding loss > 0
                model(x2, a, track=2)
                
                #backward pass
                model.optimize()
                for key in model.losses: cum_losses[key] += model.losses[key]  
        
        epoch_time = time.time() - epoch_start_time

        #optionally average the accumulated losses for the epoch
        #for key in losses: cum_losses[key] /= len(MSAdata_loader)
        total_loss = model.get_total_loss(cum_losses)

        if model_config['name'] == 'ProtTP':
            print ("Epoch[{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Attribute Loss: {:.4f} --- {:.3f} seconds" 
                .format(epoch+1, train_config['num_epochs'], cum_losses['reconst_loss'], cum_losses['kl_div'], cum_losses['attr_loss'], epoch_time))
        if model_config['name'] == 'MSATP':
            print ("Epoch[{}/{}], Reconst Loss (MSA): {:.4f}, Reconst Loss (Mutants): {:.4f}, KL Div (MSA): {:.4f}, KL Div (Mutants): {:.4f}, Attr Loss: {:.4f} --- {:.3f} seconds" 
                .format(epoch+1, train_config['num_epochs'], cum_losses['reconst_loss1'], cum_losses['reconst_loss2'], cum_losses['kl_div1'], cum_losses['kl_div2'], cum_losses['attr_loss'], epoch_time)) 
            
        #update the best model after each epoch
        if epoch == 0 or total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), save_path + '/best.pth')
            print('Best model saved')
    
    total_time = time.time() - start_time
    print('Total Time: {:.3f} seconds'.format(total_time))   

