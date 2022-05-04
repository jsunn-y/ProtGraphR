
import argparse
import json
import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from src.load_dataset import load_dataset
from src.model import *
from src.encoding_utils import *

def run_forward(model, dataset, device, train_config):
    model.eval()
    with torch.no_grad():
        X = dataset.X.to(device)
        if model.variational:
            mu, log_var = model.encode(X)
            save_embeddings = mu
            
            #this was wrong
            #embeddings = model.reparameterize(mu, log_var)
        else:
            save_embeddings = model.encode(X)
        
        save_reconstructions = model.decode(save_embeddings).cpu()
        save_embeddings = save_embeddings.cpu()

    # loader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False)

    # save_embeddings = np.array([])
    # save_reconstructions = np.array([])
    # #run 1 epoch forward in batches, not necessary, but may be better for memory
    # for i, all in enumerate(loader):
    #     model.eval()

    #     with torch.no_grad():
    #         #only take the x values
    #         if len(all) == 2:
    #             x = all[0] 
    #         else:
    #             x = all
    #         x = x.to(device)

    #         if model.variational:
    #             mu, log_var = model.encode(x)
    #             embedding = mu
    #             #this was wrong
    #             #embeddings = model.reparameterize(mu, log_var)
    #         else:
    #             embedding = model.encode(x)
            
    #         reconstructions = model.decode(embedding).cpu()

    #         if save_embeddings.shape[0] == 0:
    #             save_embeddings = embedding.cpu()
    #             save_reconstructions = reconstructions.cpu()
    #         else:
    #             save_embeddings = np.concatenate([save_embeddings, embedding.cpu()], axis=0)
    #             save_reconstructions = np.concatenate([save_reconstructions, reconstructions.cpu()], axis=0)
            
    reconst_seqs =  reconst2seq(save_reconstructions)
    return save_embeddings, reconst_seqs, save_reconstructions

def extract_features(save_path, data_config, model_config, train_config, device):
    '''Saves features after training. '''
    print('#################### Feature Extraction ####################')

    # Get model class and load data
    model_class = get_model_class(model_config['name'])
    
    if model_config['name'] == 'ProtTP':
        dataset = load_dataset(data_config, model_config, extract=True)
        model = model_class(model_config=model_config, dataset=dataset).to(device)

    #Get embeddings
    model.load_state_dict(torch.load(save_path + '/best.pth'))

    embeddings, reconst_seqs, reconstructions = run_forward(model, dataset, device, train_config)

    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
    print("Saved Features: " + os.path.join(save_path, 'embeddings.npy'))
    #np.save(os.path.join(save_path, 'reconstructions.npy'), reconstructions.cpu())
    reconst_seqs.to_csv(os.path.join(save_path, 'reconstructions.csv'))
    print("Saved Reconstructions: " + os.path.join(save_path, 'reconstructions.csv'))

    #get the log probabilities
    softmax = torch.nn.Softmax(dim=2)
    X_reconstructions = softmax(unflatten(torch.tensor(reconstructions)))
    X = unflatten(torch.tensor(dataset.X))
    probs = []
    #contains all of the likelihoods for each AA
    save_likelihoods = np.array([])

    for x, x_reconst in zip(X, X_reconstructions):
        #print(torch.sum(x, axis = 1))
        prob, likelihoods = compute_log_probability(x,x_reconst)
        probs.append(prob)
        #print(likelihoods.shape)
        if save_likelihoods.shape[0] == 0:
            save_likelihoods = likelihoods
        else:
            save_likelihoods = np.concatenate([save_likelihoods, likelihoods], axis=0)

    print(save_likelihoods.shape)
    probs = pd.DataFrame(probs, columns=['log_prob'])
    probs.to_csv(os.path.join(save_path, 'log_probs.csv'))
    print("Saved Probabilities: " + os.path.join(save_path, 'log_probs.csv'))
    np.save(os.path.join(save_path, 'likelihoods.npy'), save_likelihoods)
    print("Saved Likelihoods: " + os.path.join(save_path, 'likelihoods.npy'))

    #perform extraction on the MSAs
    if model_config['name'] == 'MSATP':
        MSAembeddings, MSAreconst_seqs, MSAreconstructions = run_forward(model, MSAdataset, device, train_config)    

        np.save(os.path.join(save_path, 'MSAembeddings.npy'), MSAembeddings)
        print("Saved Features: " + os.path.join(save_path, 'MSAembeddings.npy'))
        #np.save(os.path.join(save_path, 'MSAreconstructions.npy'), reconstructions.cpu())
        MSAreconst_seqs.to_csv(os.path.join(save_path, 'MSAreconstructions.csv'))
        print("Saved Reconstructions: " + os.path.join(save_path, 'MSAreconstructions.csv'))

    #for debugging
    #print(reconst_seqs.iloc[-1,0])
    #print(MSAreconst_seqs.iloc[0,0])

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

