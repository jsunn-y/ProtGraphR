import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from util.encoding_utils import *

encoding_dict = {
    'one-hot' : generate_onehot,
    'georgiev' : generate_georgiev,                
}

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

class PABPDataset(BaseDataset):
    """Class for PABP-specific datasets."""
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.all_combos = self.data["seq"].apply(self.trim)
        self.n_positions_combined = len(self.all_combos[0])
        self.y = self.data["log_fitness"].values

    @staticmethod
    def trim(sequence):
        return sequence[8:-6]

    @staticmethod
    def cut(sequence):
        return sequence[122:204]

class MSADataset(Dataset):
    """Separate Class for processing MSAs."""
    def __init__(self, dataframe, MSAdataframe, encoding):
        
        self.data = MSAdataframe
        self.encoding = encoding
        self.all_combos = self.data["seq"].apply(self.trim)
        self.n_positions_combined = len(self.all_combos[0])

        #make this better later
        self.attributes = torch.zeros(len(self.all_combos))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):            
        return self.X[index]

    def encode_X(self):
        self.X = torch.tensor(encoding_dict[self.encoding](self.all_combos.values)).float()
        self.input_dim = self.X.shape[1]

    @staticmethod
    def trim(sequence):
        return sequence[8:-6]

    @staticmethod
    def pad(sequence):
        return '........' + sequence[8:-6] + '......'