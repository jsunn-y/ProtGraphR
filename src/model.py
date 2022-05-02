import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GAE, VGAE, GCN, GATv2Conv, GCNConv, NNConv, global_mean_pool

from src.loss import *

def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
    
class GraphEncoder(nn.Module):
    def __init__(self, model_config):
        """
        """
        super(GraphEncoder, self).__init__()

        # save all of the info
        self.node_dim = model_config['node_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['dropout']
        self.edge_dim = model_config['edge_dim']

        # a list of GATv2 layers, with dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(self.num_layers):
            layer = pyg_nn.GATv2Conv(in_channels=self.hidden_dim,
                                     out_channels=self.hidden_dim,
                                     dropout=self.dropout,
                                     edge_dim = self.edge_dim)
            self.convs.append(layer)
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        # fully-connected final layer
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        """
        Args
        - data: pyg.data.Batch, a batch of graphs

        Returns: torch.Tensor, shape [batch_size], unnormalized classification
            probability for each graph
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch) #use edge_attr.to(torch.float32)

        for l, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if l != self.num_layers - 1:
                x = self.bns[l](x)
                x = F.relu(x)

        x = pyg_nn.global_mean_pool(x, batch=batch)
        x = self.fc(x)
        return x

# class GAE_new(GAE):
#     def __init__(self):
#         super(GAE_new, self).__init__()

model_dict = {
    'GAE' : GAE,
    'VGAE' : VGAE              
}