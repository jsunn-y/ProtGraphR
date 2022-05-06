import pandas as pd
from pyparsing import col
from torch.utils.data import DataLoader
from src.Datasets import *
from Bio import AlignIO

#could make this better (some should go in config file, some should be parser arguments for evSeq)
# data_dict = {
#     'GB1' : {'dataset_class': GB1Dataset, 'train_file': './data/GB1_AllPreds.csv'},
#     'PABP' : {'dataset_class': PABPDataset, 'train_file': './data/PABP_AllPreds.csv', 'alignment_file': './data/PABP_YEAST.a2m', 'extract_file': './data/PABP_AllPreds.csv'} #'./data/All_DSM_PABP.csv'
# }

#default is training, not extracting
def load_dataset(data_config, model_config, extract=False):

    dataset = GraphDataset('data/' + data_config['name'])
    
    #get the node and edge dimension from the first graph
    data = dataset[0]
    model_config['node_dim'] = data.num_node_features
    model_config['edge_dim'] = data.num_edge_features
    print(data)
    return dataset, model_config
    
    

