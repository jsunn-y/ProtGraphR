import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    # Initialize dataset
    dataset = load_dataset(data_config)
    # Get model class
    model_class = get_model_class(model_config['name'])
    encoding = data_config['encoding']

    # Initialize model
    model = model_class(model_config, dataset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    # Initialize dataloaders
    data_loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True)

    variational = model_config['kl_div_weight'] != 0
    attributes = model_config['attr_decoding_loss_weight'] != 0

    # Start training
    best_loss = 0
    for epoch in range(train_config['num_epochs']):
        for i, (x, a) in enumerate(data_loader):
            # Forward pass
            x = x.to(device)#.view(-1, dataset.input_dim)
            a = a.to(device)#.view(-1, input_dim)

            if variational:
                if attributes:
                    x_reconst, mu, log_var, attr_reconst = model(x)
                    attr_loss = F.mse_loss(attr_reconst, a)
                else:
                    x_reconst, mu, log_var = model(x)
                    attr_loss = 0
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 
            else:
                if attributes:
                    x_reconst, z, attr_reconst = model(x)
                    attr_loss = F.mse_loss(attr_reconst, a)
                else:
                    x_reconst, z = model(x)
                    attr_loss = 0
                kl_div = 0

            if encoding == 'one-hot':
                #reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
                loss_function = nn.CosineEmbeddingLoss(reduction='none')
                reconst_loss = loss_function(x_reconst, x, torch.ones(x.shape[0]).to(device)).sum()
            elif encoding == 'georgiev':
                loss_function = nn.MSELoss(reduction='none')
                reconst_loss = loss_function(x_reconst, x).sum()

            loss = reconst_loss*model_config["reconstruction_loss_weight"] + kl_div*model_config["kl_div_weight"] + attr_loss*model_config["attr_decoding_loss_weight"]
            
            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print ("Epoch[{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Attribute Loss: {:.4f}" 
            .format(epoch+1, train_config['num_epochs'], reconst_loss, kl_div, attr_loss))
        
        #update the best model after each epoch
        if epoch == 0 or loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path + '/best.pth')
            print('Best model saved')   

