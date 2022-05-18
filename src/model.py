import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCN, GATv2Conv, GCNConv, NNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import (
    add_self_loops,
    negative_sampling,
    remove_self_loops,
)

#from ..inits import reset

def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
    
class GraphEncoder(nn.Module):
    def __init__(self, model_config):
        """
        Encoder module for the GAE or VGAE.
        """
        super(GraphEncoder, self).__init__()

        # save all of the info
        self.node_dim = model_config['node_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['dropout']
        self.edge_dim = model_config['edge_dim']

        self.variational = model_config['kl_div_weight'] != 0

        self.conv1 = GATv2Conv(
            in_channels=self.node_dim,
            out_channels=self.hidden_dim,
            dropout=self.dropout,
            edge_dim=self.edge_dim
        )
        
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        self.convs = nn.ModuleList()
        
        for l in range(self.num_layers-1):
            layer = GATv2Conv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            dropout=self.dropout,
            edge_dim=self.edge_dim
            )
            self.convs.append(layer)
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        
        self.convvar = GATv2Conv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            dropout=self.dropout,
            edge_dim=self.edge_dim
        )

        # fully-connected final layer
        #self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, data: pyg.data.Data, pool = False) -> torch.Tensor:
        """
        Args
        - data: pyg.data.Batch, a batch of graphs

        Returns: torch.Tensor, shape [batch_size], unnormalized classification
            probability for each graph
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr.to(torch.float32), data.batch)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.bns[0](x)

        if self.num_layers == 2 and self.variational:
                var = self.convvar(x, edge_index, edge_attr)

        for l, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if l != self.num_layers - 2:    
                #should be no activation in the final layer
                x = self.bns[l+1](x)
                x = F.elu(x)
            if l == self.num_layers - 3:
                if self.variational:
                    var = self.convvar(x, edge_index, edge_attr)
    
        if pool:
            z = pyg_nn.global_max_pool(x, batch=batch)
            #z = self.fc(z)
            return z

        if self.variational:
            z = x, var
            return z
        else:
            return x

#written by pytorch below
EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

#write the classes for other self-supervised decoders here

class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, model_config, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder

        #GAE.reset_parameters(self)

    # def reset_parameters(self):
    #     reset(self.encoder)
    #     reset(self.decoder)


    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)


    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)


    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
    
    #the methods below are not in the original paper:


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, model_config, decoder=None):
        super().__init__(encoder, model_config, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, extract=False, *args, **kwargs):
        """"""
        if extract == False:
            self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
            self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
            z = self.reparametrize(self.__mu__, self.__logstd__)
        else:
            #self.__mu__ = self.encoder(pool = True, *args, **kwargs)
            self.__mu__, self.__logstd__ = self.encoder(pool = False, *args, **kwargs)
            z = self.__mu__
            
        return z


    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    
    #the methods below are not in the original paper:


model_dict = {
    'GAE' : GAE,
    'VGAE' : VGAE              
}