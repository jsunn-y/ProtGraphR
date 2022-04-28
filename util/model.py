import torch
import torch.nn as nn
import torch.nn.functional as F

from util.loss import *

def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError

class BaseModel(nn.Module):
    def __init__(self, model_config, dataset):
        #not sure what this line does tbh
        super(BaseModel, self).__init__()
        
        dropout = model_config['dropout']
        enc_dim1 = model_config['enc_dim1']
        enc_dim2 = model_config['enc_dim2']
        z_dim = model_config['z_dim']
        dec_dim1 = model_config['dec_dim1']
        dec_dim2 = model_config['dec_dim2']
        attr_dim = dataset.attributes.shape[1]

        input_dim = dataset.input_dim
        self.encoding = dataset.encoding
        self.variational = model_config['kl_div_weight'] != 0
        self.attributes = model_config['attr_decoding_loss_weight'] != 0
        
        #if decoding attributes, check to make sure attributes were specified in the dataset
        if self.attributes:
            assert torch.sum(dataset.attributes, 0)[0] != 0

        self.losses = {}
        self.losses['reconst_loss'] = 0
        self.losses['kl_div'] = 0
        self.losses['attr_loss'] = 0

        self.reconst_loss_weight = model_config["reconstruction_loss_weight"]
        self.attr_loss_weight = model_config["attr_decoding_loss_weight"]
        self.kl_div_weight = model_config["kl_div_weight"]

        self.dropout = nn.Dropout(dropout)
        #encoder layers
        self.fce1 = nn.Linear(input_dim, enc_dim1)
        self.fce2 = nn.Linear(enc_dim1, enc_dim2)
        self.bne = nn.BatchNorm1d(enc_dim2)
        self.fce3 = nn.Linear(enc_dim2, z_dim)
        self.fcvar = nn.Linear(enc_dim2, z_dim)
        
        #decoder layers
        self.fcd1 = nn.Linear(z_dim, dec_dim1)
        self.fcd2 = nn.Linear(dec_dim1, dec_dim2)
        self.fcd3 = nn.Linear(dec_dim2, input_dim)

        #attribute decoder layers
        self.fcad1 = nn.Linear(z_dim, z_dim)
        self.fcad2 = nn.Linear(z_dim, attr_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fce1(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fce2(h1))
        h2 = self.bne(h2)
        
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
        h1 = self.dropout(h1)
        h2 = F.relu(self.fcd2(h1))
        return self.fcd3(h2)
    
    def decode_attr(self, z):
        h1 = F.relu(self.fcad1(z))
        return self.fcad2(z)
    
    def init_optimizer(self, train_config):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=train_config['learning_rate'])

    def get_total_loss(self, losses):
        return losses['reconst_loss']*self.reconst_loss_weight + losses['kl_div']*self.kl_div_weight + losses['attr_loss']*self.attr_loss_weight

    def optimize(self):
        self.optimizer.zero_grad()
        self.total_loss = self.get_total_loss(self.losses)
        self.total_loss.backward()
        self.optimizer.step()

class ProtTP(BaseModel):
    def __init__(self, **kwargs):
        #super(ProtTP, self).__init__()
        super().__init__(**kwargs)

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
        #return self.losses

class MSATP(BaseModel):
    def __init__(self, MSAdataset, **kwargs):
        super().__init__(**kwargs)

        #separate kl_divergences for each dataset
        self.losses['kl_div1'] = 0
        self.losses['kl_div2'] = 0

        #separate the reconstruction losses for each dataset
        self.losses['reconst_loss1'] = 0
        self.losses['reconst_loss2'] = 0

    def forward(self, x, a, track):
        #both tracks initially encode the sequence to the latent space
        if self.variational:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            #might not need to calculate kl_dov on second pass
            if track == 1:
                self.losses['kl_div'] = 0
                self.losses['kl_div1'] = compute_kl_div(mu, log_var)
                self.losses['kl_div'] += self.losses['kl_div1']
            if track == 2:
                self.losses['kl_div2'] = compute_kl_div(mu, log_var)
                self.losses['kl_div'] += self.losses['kl_div2'] 
        else:
            z = self.encode(x)
        
        #track 1 decodes the MSA
        if track == 1:
            self.losses['reconst_loss'] = 0
            x_reconst = self.decode(z)
            self.losses['reconst_loss1'] = compute_reconst_loss(self.encoding, x, x_reconst)
            self.losses['reconst_loss'] += self.losses['reconst_loss1']
        
        #track 2 decodes the attributes from the mutant sequences
        elif track == 2:
            if self.attributes:
                attr_reconst = self.decode_attr(z)
                self.losses['attr_loss'] = compute_attr_loss(a, attr_reconst)

            #reconstruct the mutants
            x_reconst = self.decode(z)
            self.losses['reconst_loss2'] = compute_reconst_loss(self.encoding, x, x_reconst)
            self.losses['reconst_loss'] += 10 * self.losses['reconst_loss2']

model_dict = {
    'ProtTP' : ProtTP,
    'MSATP' : MSATP,                
}