import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.encoding_utils import *
import torch_geometric
from tqdm.auto import tqdm

encoding_dict = {
    'one-hot' : generate_onehot,
    'georgiev' : generate_georgiev,                
}

class GraphDataset(Dataset):
    def __init__(self, graph_dir: str = ""):
        """
        Generates a dataset of pytorch graph objects. Requires generating and preprocessing the graphs before usage.
        - graph_dir: str, path to folder of .pt files containing the pytorch geomentric graph files
        """
        super().__init__()
        
        self._graph_dir = graph_dir
        self._s2g_list = os.listdir(graph_dir)
        
        # #stuff below is for loading all the data initially
        # self.pygs = []
        # pbar = tqdm()
        # pbar.set_description('Loading Graphs')
        # pbar.reset(total=len(self._s2g_list))

        # for s2g in self._s2g_list:
        #     s2g = os.path.join(os.getcwd(), graph_dir, s2g) 
        #     self.pygs.append(torch.load(s2g))
        #     pbar.update()

    def __len__(self) -> int:
        return len(self._s2g_list)

    def __getitem__(self, index: int):
        s2g = os.path.join(os.getcwd(), self._graph_dir, self._s2g_list[index]) 
        return torch.load(s2g)

        # return self.pygs[index]

class BaseDataset(Dataset):
    """Base class for labeled datasets."""

    def __init__(self, dataframe, encoding, attribute_names, rank_attributes=True):

        self.data = dataframe
        self.encoding = encoding
        self.attribute_names = attribute_names
        self.N = len(self.data)

        if self.attribute_names:
            if rank_attributes:
                for attribute_name in attribute_names:
                    self.data[attribute_name] = self.data[attribute_name].rank()
            self.attributes = self.data[attribute_names].values
            scaler = StandardScaler()
            self.attributes = scaler.fit_transform(self.attributes)
            self.attributes = torch.tensor(self.attributes).float()
        else:
            #placeholder
            self.attributes = torch.zeros((self.N, 1))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):          
        return self.X[index], self.attributes[index]

    def encode_X(self):
        #all_combos needs to be defined from the child classes
        self.X = torch.tensor(encoding_dict[self.encoding](self.all_combos.values)).float()
        self.input_dim = self.X.shape[1]
        self.n_residues = self.input_dim/len(ALL_AAS)


class GB1Dataset(BaseDataset):
    """Class for GB1-specific datasets."""
    def __init__(self, full_sequence = False, SD_only = False, **kwargs):
        
        super().__init__(**kwargs)
        
        self.data['num_muts'] = self.data['Combo'].apply(self.diff_letters)

        #only select variants with 2 or less mutations
        if SD_only:
            self.data = self.data[self.data['num_muts'] <= 2]
        
        self.all_combos = self.data["Combo"]
        
        if full_sequence:
            self.all_combos = self.all_combos.apply(self.generate_full)
        self.n_positions_combined = len(self.all_combos[0])
        #self.y = self.data["Fitness"].values
    
    @staticmethod
    def diff_letters(a, b = 'VDGV'):
        return sum ( a[i] != b[i] for i in range(len(a)) )

    @staticmethod
    def generate_full(combo):
        seq = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
        seq = seq[:38] + combo[0] + combo[1] + combo[2] + seq[41:53] + combo[3] + seq[54:]
        return seq