import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.encoding_utils import *


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
        self.pygs = []
        for s2g in self._s2g_list:
            self.pygs.append(torch.load(s2g))

    def __len__(self) -> int:
        return len(self._s2g_list)

    def __getitem__(self, index: int) -> torch_geometric.Data:
        return self.pygs[index]