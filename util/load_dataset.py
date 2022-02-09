import pandas as pd
from torch.utils.data import DataLoader
from util.Datasets import *

data_dict = {
    'GB1' : ('./data/GB1_AllPreds.csv', GB1Dataset),
    'PABP' : ('./data/PABP_AllPreds.csv', PABPDataset)                
}

def load_dataset(data_config):
    dataset_name = data_config['name']
    df = pd.read_csv(data_dict[dataset_name][0])
    dataset = data_dict[dataset_name][1](dataframe = df, encoding = data_config['encoding'], attribute_names = data_config['attributes'])
    
    if data_config['encoding'] == 'one-hot':
        dataset.generate_onehot()
    elif data_config['encoding'] == 'georgiev':
        dataset.generate_georgiev()
    return dataset

