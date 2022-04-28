import pandas as pd
from pyparsing import col
from torch.utils.data import DataLoader
from util.Datasets import *
from Bio import AlignIO

#could make this better (some should go in config file, some should be parser arguments for evSeq)
data_dict = {
    'GB1' : {'dataset_class': GB1Dataset, 'train_file': './data/GB1_AllPreds.csv'},
    'PABP' : {'dataset_class': PABPDataset, 'train_file': './data/PABP_AllPreds.csv', 'alignment_file': './data/PABP_YEAST.a2m', 'extract_file': './data/PABP_AllPreds.csv'} #'./data/All_DSM_PABP.csv'
}

#default is training, not extracting
def load_dataset(data_config, model_config, extract=False):
    dataset_name = data_config['name']
    model_name = model_config['name']

    if extract:
        df = pd.read_csv(data_dict[dataset_name]['extract_file'])
    else:
        df = pd.read_csv(data_dict[dataset_name]['train_file'])

    dataset = data_dict[dataset_name]['dataset_class'](dataframe = df, encoding = data_config['encoding'], attribute_names = data_config['attributes'])
    dataset.encode_X()

    if model_name == 'MSATP':
        align = AlignIO.read(data_dict[dataset_name]['alignment_file'], format='fasta')
        list = []
        for entry in align:
            list.append(str(entry.seq))
        MSAdf = pd.DataFrame(list, columns=['seq'])
        # Sam sinai only used the first part of the alignment
        # MSAdf = MSAdf[:50000]
        MSAdf.to_csv('MSA.csv')
        MSAdataset = MSADataset(dataframe = df, MSAdataframe = MSAdf, encoding = data_config['encoding'])
        MSAdataset.encode_X()
        return dataset, MSAdataset
    elif model_name == "ProtTP":
        return dataset
    
    

