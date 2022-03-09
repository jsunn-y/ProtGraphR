
import argparse
import json
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from util.load_dataset import load_dataset
from util.model import *
from util.encoding_utils import *

def index2AA(index):
    return ALL_AAS[index]

index2AA = np.vectorize(index2AA)

def compute_log_probability(x,x_reconst):
    #flipped from sam sinai code
    prod_mat= x @ x_reconst.T
    #take the log before or after?
    prod_mat = np.log(prod_mat)
    sum_diag=np.trace(prod_mat)
    return sum_diag

def reconst2seq(reconstructions):
    """Converts a one-hot encoding of a reconstruction to its corresponding amino-acid sequence."""
    reconstructions = unflatten(torch.tensor(reconstructions))
    reconstructions = torch.argmax(reconstructions, axis = 2)
    letters = index2AA(reconstructions)
    seqs = []
    for row in letters:
        seqs.append(''.join(row))
    return pd.DataFrame(seqs)

def run_forward(model, dataset, device):
    with torch.no_grad():
        X = dataset.X.to(device)
        if model.variational:
            mu, log_var = model.encode(X)
            embeddings = mu
            #this was wrong
            #embeddings = model.reparameterize(mu, log_var)
        else:
            embeddings = model.encode(X)
        
        reconstructions = model.decode(embeddings).cpu()
        reconst_seqs =  reconst2seq(reconstructions)
    return embeddings, reconst_seqs, reconstructions

def extract_features(save_path, data_config, model_config, train_config, device):
    '''Saves features after training. '''
    print('#################### Feature Extraction ####################')

    # Get model class and load data
    model_class = get_model_class(model_config['name'])
    if model_config['name'] == 'ProtTP':
        dataset = load_dataset(data_config, model_config, extract=True)
        model = model_class(model_config=model_config, dataset=dataset).to(device)
    elif model_config['name'] == 'MSATP':
        dataset, MSAdataset = load_dataset(data_config, model_config, extract=True)
        model = model_class(model_config=model_config, dataset=dataset, MSAdataset=MSAdataset).to(device)

    #Get embeddings
    model.load_state_dict(torch.load(save_path + '/best.pth'))

    embeddings, reconst_seqs, reconstructions = run_forward(model, dataset, device)
    softmax = torch.nn.Softmax(dim=2)
    reconstructions = softmax(unflatten(torch.tensor(reconstructions)))
    X = unflatten(torch.tensor(dataset.X))

    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings.cpu())
    print("Saved Features: " + os.path.join(save_path, 'embeddings.npy'))
    #np.save(os.path.join(save_path, 'reconstructions.npy'), reconstructions.cpu())
    reconst_seqs.to_csv(os.path.join(save_path, 'reconstructions.csv'))
    print("Saved Reconstructions: " + os.path.join(save_path, 'reconstructions.csv'))

    #get the log probabilities
    probs = []
    for x, x_reconst in zip(X, reconstructions):
        #print(torch.sum(x, axis = 1))
        probs.append(compute_log_probability(x,x_reconst))
    probs = pd.DataFrame(probs, columns=['log_prob'])
    probs.to_csv(os.path.join(save_path, 'log_probs.csv'))
    print("Saved Probabilities: " + os.path.join(save_path, 'log_probs.csv'))

    #perform extraction on the MSAs
    if model_config['name'] == 'MSATP':
        MSAembeddings, MSAreconst_seqs, MSAreconstructions = run_forward(model, MSAdataset, device)    

        np.save(os.path.join(save_path, 'MSAembeddings.npy'), MSAembeddings.cpu())
        print("Saved Features: " + os.path.join(save_path, 'MSAembeddings.npy'))
        #np.save(os.path.join(save_path, 'MSAreconstructions.npy'), reconstructions.cpu())
        MSAreconst_seqs.to_csv(os.path.join(save_path, 'MSAreconstructions.csv'))
        print("Saved Reconstructions: " + os.path.join(save_path, 'MSAreconstructions.csv'))


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

