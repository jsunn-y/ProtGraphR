
import argparse
import json
import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from src.load_dataset import load_dataset
from src.train import eval
from src.model import *
from src.encoding_utils import *

def extract_features(save_path, data_config, model_config, train_config, device):
    '''Saves features after training. '''
    print('#################### Feature Extraction ####################')

    # Get model class and load data
    model_class = get_model_class(model_config['name'])
    dataset, model_config = load_dataset(data_config, model_config, extract=True)
    model = model_class(GraphEncoder(model_config = model_config)).to(device)
    
    #Initialize dataloaders
    loader = DataLoader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=True)

    #Get best model
    model.load_state_dict(torch.load(save_path + '/best.pth'))
    pbar = tqdm()
    embeddings = eval(model, device, loader, pbar)

    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
    print("Saved Features: " + os.path.join(save_path, 'embeddings.npy'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        required=True,
                        help='config file for experiments')
    parser.add_argument('--exp_name', type=str,
                        required=False, default='',
                        help='experiment name (default will be config folder name)')
    parser.add_argument('-d', '--device', type=int,
                    required=False, default=0,
                    help='device to run the experiment on')
    args = parser.parse_args()    

    # Get experiment name
    exp_name = args.exp_name if len(args.exp_name) > 0 else args.config_file[:-5]

    # Get JSON config file
    config_file = os.path.join(os.getcwd(), 'saved', exp_name, args.config_file)
    
    # Load JSON config file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Get save directory
    save_dir = os.path.join(os.getcwd(), 'saved', exp_name)

    # Get device ID
    if torch.cuda.is_available() and args.device >= 0:
        assert args.device < torch.cuda.device_count()
        device = 'cuda:{:d}'.format(args.device)
    else:
        device = 'cpu'
    print('Device:\t {}'.format(device))

    extract_features(
        save_path=save_dir,
        data_config=config['data_config'],
        model_config=config['model_config'],
        train_config=config['train_config'],
        device=device
    )

