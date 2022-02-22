import pandas as pd
from pyparsing import col
from torch.utils.data import DataLoader
from util.Datasets import *
from Bio import AlignIO

data_dict = {
    'GB1' : ('./data/GB1_AllPreds.csv', GB1Dataset, ),
    'PABP' : ('./data/PABP_AllPreds.csv', PABPDataset, './data/PABP_YEAST.a2m'),      #'./data/All_DSM_PABP.csv'     
}

def load_dataset(data_config, model_config):
    dataset_name = data_config['name']
    model_name = model_config['name']

    df = pd.read_csv(data_dict[dataset_name][0])
    dataset = data_dict[dataset_name][1](dataframe = df, encoding = data_config['encoding'], attribute_names = data_config['attributes'])
    dataset.encode_X()
    if model_name == 'MSATP':
        align = AlignIO.read(data_dict[dataset_name][2], format='fasta')
        list = []
        for entry in align:
            list.append(str(entry.seq))
        MSAdf = pd.DataFrame(list, columns=['seq'])
        MSAdf.to_csv('MSA.csv')
        ZSdf = pd.read_csv(data_dict[dataset_name][0])
        MSAdataset = MSADataset(dataframe = df, MSAdataframe = MSAdf, encoding = data_config['encoding'])
        MSAdataset.encode_X()
        return dataset, MSAdataset
    elif model_name == "ProtTP":
        return dataset
    
    

