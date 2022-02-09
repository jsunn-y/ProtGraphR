import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from util.georgiev_utils import *

ALL_AAS = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")

class BaseDataset(Dataset):
    """Base class for Datasets."""

    def __init__(self, dataframe, encoding, attribute_names, rank_attributes=True):

        self.data = dataframe
        self.encoding = encoding
        self.attribute_names = attribute_names

        if self.attribute_names:
            if rank_attributes:
                for attribute_name in attribute_names:
                    self.data[attribute_name] = self.data[attribute_name].rank()
            self.attributes = self.data[attribute_names].values
            scaler = StandardScaler()
            self.attributes = scaler.fit_transform(self.attributes)
            self.attributes = torch.tensor(self.attributes).float()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):            
        return self.X[index], self.attributes[index]

    def generate_onehot(self):
        """
        Builds a onehot encoding for a given combinatorial space.
        """
        # Make a dictionary that links amino acid to index
        one_hot_dict = {aa: i for i, aa in enumerate(ALL_AAS)}
    
        # Build an array of zeros
        onehot_array = np.zeros([len(self.all_combos), self.n_positions_combined, 20])
        
        # Loop over all combos. This should all be vectorized at some point.
        for i, combo in enumerate(self.all_combos):
            
            # Loop over the combo and add ones as appropriate
            for j, character in enumerate(combo):
                
                # Add a 1 to the appropriate position
                onehot_ind = one_hot_dict[character]
                onehot_array[i, j, onehot_ind] = 1
                
        # Return the flattened array
        self.X = torch.tensor(onehot_array.reshape(onehot_array.shape[0],-1)).float()
        self.input_dim = self.n_positions_combined*20 
        return 
    
    def generate_georgiev(self):
        """
        Builds a georgiev encoding for a given combinatorial space.
        """
        self.X = torch.tensor(seqs_to_georgiev(self.all_combos)).float()
        self.input_dim = self.n_positions_combined*19 
        return 

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
        

        self.y = self.data["Fitness"].values
    
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
        
        self.all_combos = self.data["seq"]#.apply(self.cut)
        self.n_positions_combined = len(self.all_combos[0])
        self.y = self.data["log_fitness"].values

    @staticmethod
    def cut(sequence):
        return sequence[122:204]