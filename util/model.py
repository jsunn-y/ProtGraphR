import torch
import torch.nn as nn
import torch.nn.functional as F

from util.loss import *

def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
        
class ProtTP(nn.Module):
    def __init__(self, model_config, dataset):
        super(ProtTP, self).__init__()

        h_dim1 = model_config['h_dim1']
        h_dim2 = model_config['h_dim2']
        z_dim = model_config['z_dim']
        attr_dim = dataset.attributes.shape[1]
        
        input_dim = dataset.input_dim
        self.variational = model_config['kl_div_weight'] != 0
        self.attributes = model_config['attr_decoding_loss_weight'] != 0
        self.encoding = dataset.encoding

        self.losses = {}
        self.losses['reconst_loss'] = 0
        self.losses['kl_div'] = 0
        self.losses['attr_loss'] = 0

        self.reconst_loss_weight = model_config["reconstruction_loss_weight"]
        self.attr_loss_weight = model_config["attr_decoding_loss_weight"]
        self.kl_div_weight = model_config["kl_div_weight"]

        #encoder layers
        self.fce1 = nn.Linear(input_dim, h_dim1)
        self.fce2 = nn.Linear(h_dim1, h_dim2)
        self.fce3 = nn.Linear(h_dim2, z_dim)
        self.fcvar = nn.Linear(h_dim2, z_dim)
        
        #decoder layers
        self.fcd1 = nn.Linear(z_dim, h_dim2)
        self.fcd2 = nn.Linear(h_dim2, h_dim1)
        self.fcd3 = nn.Linear(h_dim1, input_dim)

        #attribute decoder layers
        self.fcad1 = nn.Linear(z_dim, z_dim)
        self.fcad2 = nn.Linear(z_dim, attr_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fce1(x))
        h2 = F.relu(self.fce2(h1))
        if self.variational:
            return self.fce3(h2), self.fcvar(h2)
        else:
            return self.fce3(h2)
    
    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, -16, 16)
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = F.relu(self.fcd1(z))
        h2 = F.relu(self.fcd2(h1))
        return self.fcd3(h2)
        
    def decode_attr(self, z):
        h1 = F.relu(self.fcad1(z))
        return self.fcad2(z)

    def forward(self, x, a):
        if self.variational:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_reconst = self.decode(z)
            self.losses['kl_div'] = compute_kl_div(mu, log_var)

            if self.attributes:
                attr_reconst = self.decode_attr(z)
                self.losses['attr_loss'] = compute_attr_loss(a, attr_reconst)
        else:
            z = self.encode(x)
            x_reconst = self.decode(z)
            if self.attributes:
                attr_reconst = self.decode_attr(z)
                self.losses['attr_loss'] = compute_attr_loss(a, attr_reconst)
        
        self.losses['reconst_loss'] = compute_reconst_loss(self.encoding, x, x_reconst)
        return self.losses

    def init_optimizer(self, train_config):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=train_config['learning_rate'])

    def optimize(self):
        self.total_loss = self.losses['reconst_loss']*self.reconst_loss_weight + self.losses['kl_div']*self.kl_div_weight + self.losses['attr_loss']*self.attr_loss_weight
        
        self.optimizer.zero_grad()
        self.total_loss.backward()
        self.optimizer.step()


class MSATP(nn.Module):
    def __init__(self, model_config, dataset, MSAdataset):
        super(MSATP, self).__init__()
        
        h_dim1 = model_config['h_dim1']
        h_dim2 = model_config['h_dim2']
        z_dim = model_config['z_dim']

        input_dim = MSAdataset.input_dim
        self.variational = model_config['kl_div_weight'] != 0
        self.attributes = model_config['attr_decoding_loss_weight'] != 0
        self.encoding = MSAdataset.encoding

        self.losses = {}
        self.losses['reconst_loss'] = 0
        self.losses['kl_div'] = 0
        self.losses['attr_loss'] = 0

        self.reconst_loss_weight = model_config["reconstruction_loss_weight"]
        self.attr_loss_weight = model_config["attr_decoding_loss_weight"]
        self.kl_div_weight = model_config["kl_div_weight"]

        #encoder layers
        self.fce1 = nn.Linear(input_dim, h_dim1)
        self.fce2 = nn.Linear(h_dim1, h_dim2)
        self.fce3 = nn.Linear(h_dim2, z_dim)
        self.fcvar = nn.Linear(h_dim2, z_dim)
        
        #decoder layers
        self.fcd1 = nn.Linear(z_dim, h_dim2)
        self.fcd2 = nn.Linear(h_dim2, h_dim1)
        self.fcd3 = nn.Linear(h_dim1, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fce1(x))
        h2 = F.relu(self.fce2(h1))
        if self.variational:
            return self.fce3(h2), self.fcvar(h2)
        else:
            return self.fce3(h2)
    
    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, -16, 16)
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = F.relu(self.fcd1(z))
        h2 = F.relu(self.fcd2(h1))
        return self.fcd3(h2)

    def forward(self, x, a):
        if self.variational:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_reconst = self.decode(z)
            self.losses['kl_div'] = compute_kl_div(mu, log_var)
        else:
            z = self.encode(x)
            x_reconst = self.decode(z)
        
        self.losses['reconst_loss'] = compute_reconst_loss(self.encoding, x, x_reconst)
        return self.losses

    def init_optimizer(self, train_config):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=train_config['learning_rate'])

    def optimize(self):
        self.total_loss = self.losses['reconst_loss']*self.reconst_loss_weight + self.losses['kl_div']*self.kl_div_weight + self.losses['attr_loss']*self.attr_loss_weight
        
        self.optimizer.zero_grad()
        self.total_loss.backward()
        self.optimizer.step()

model_dict = {
    'ProtTP' : ProtTP,
    'MSATP' : MSATP,                
}