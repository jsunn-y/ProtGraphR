import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.encoding_utils import *

def unflatten(tensor):
    batch_size = tensor.shape[0]
    return torch.reshape(tensor, (batch_size, -1, num_tokens))

def compute_reconst_loss(encoding, x, x_reconst):
    if encoding == 'one-hot':
        #reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        #loss_function = nn.CosineEmbeddingLoss(reduction='none')
        #reconst_loss = loss_function(x_reconst, x, torch.ones(x.shape[0]).to(device)).sum()
        loss_function = nn.CrossEntropyLoss()
        x = unflatten(x)
        x = torch.argmax(x, axis = 2)
        x_reconst = unflatten(x_reconst)
        x_reconst = torch.swapaxes(x_reconst,1,2)
        reconst_loss = loss_function(x_reconst, x)
    elif encoding == 'georgiev':
        loss_function = nn.MSELoss(reduction='none')
        reconst_loss = loss_function(x_reconst, x).sum()
    return reconst_loss

def compute_attr_loss(a, a_reconst):
    loss_function = nn.MSELoss(reduction='none')
    return loss_function(a_reconst, a).sum()

def compute_kl_div(mu, log_var):
    return - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 