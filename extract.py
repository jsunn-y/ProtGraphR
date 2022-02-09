
import argparse
import json
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from util.load_dataset import load_dataset
from util.model import *

def extract_features(save_path, data_config, model_config, train_config, device):
    '''Saves features after training. '''
    print('#################### Feature Extraction ####################')

    # Initialize dataset
    dataset = load_dataset(data_config)
    # Get model class
    model_class = get_model_class(model_config['name'])
        
    # Initialize model
    model = model_class(model_config, dataset).to(device)

    #Get embeddings
    model.load_state_dict(torch.load(save_path + '/best.pth'))

    variational = model_config['kl_div_weight'] != 0
    attributes = model_config['attr_decoding_loss_weight'] != 0

    with torch.no_grad():
        X = dataset.X.to(device)
        if variational:
            mu, log_var = model.encode(X)
            embeddings = model.reparameterize(mu, log_var)
        else:
            embeddings = model.encode(X)
        reconstructions = model.decode(embeddings)

        np.save(os.path.join(save_path, 'embeddings.npy'), embeddings.cpu())
        print("Saved Features: " + os.path.join(save_path, 'embeddings.npy'))
        np.save(os.path.join(save_path, 'reconstructions.npy'), reconstructions.cpu())
        print("Saved Reconstructions: " + os.path.join(save_path, 'reconstructions.npy'))


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

    # Get JSON config file
    config_file = os.path.join(os.getcwd(), 'configs', args.config_file)
    
    # Load JSON config file
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get experiment name
    exp_name = args.exp_name if len(args.exp_name) > 0 else args.config_file[:-5]

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

