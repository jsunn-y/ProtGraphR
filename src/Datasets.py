import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.encoding_utils import *
import torch_geometric
from tqdm.auto import tqdm

# encoding_dict = {
#     'one-hot' : generate_onehot,
#     'georgiev' : generate_georgiev,                
# }

class GraphDataset(Dataset):
    def __init__(self, graph_dir: str = ""):
        """
        Generates a dataset of pytorch graph objects. Requires generating and preprocessing the graphs
        - graph_dir: str, path to folder of .pt files 
          containing the pytorch geomentric graph files
        """
        super().__init__()
        
        self._graph_dir = graph_dir
        self._s2g_list = os.listdir(graph_dir)
        
        #stuff below is for loading all the data initially
        self.pygs = []
        pbar = tqdm()
        pbar.set_description('Loading Graphs')
        pbar.reset(total=len(self._s2g_list))

        for s2g in self._s2g_list:
            s2g = os.path.join(os.getcwd(), graph_dir, s2g) 
            self.pygs.append(torch.load(s2g))
            pbar.update()

    def __len__(self) -> int:
        return len(self._s2g_list)

    def __getitem__(self, index: int):
        # s2g = os.path.join(os.getcwd(), self._graph_dir, self._s2g_list[index]) 
        # return torch.load(s2g)

        return self.pygs[index]