from __future__ import annotations

from collections.abc import Mapping
import os
import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.utils import resample
from sklearn.metrics import ndcg_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from src.Datasets import *
from src.load_dataset import load_dataset
from src.model import *


def ndcg(y_true, y_pred):
    y_true_normalized = y_true - min(y_true)
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))


def train(model_config: dict, model: nn.Module, device: torch.device,
          data_loader1: DataLoader, data_loader2: DataLoader,
          optimizer: torch.optim.Optimizer, pbar: tqdm) -> float:
    """Trains a GNN model for one epoch.

    Args
    - model: nn.Module, GNN model, already placed on device
    - device: torch.device
    - data_loader: pyg.loader.DataLoader
    - optimizer: torch.optim.Optimizer
    - loss_fn: nn.Module

    Returns: loss
    - loss: float, avg loss across epoch
    """
    model.train()
    total_recon_loss = 0
    total_kl_div = 0
    total_zs_loss = 0

    n = len(data_loader1)
    pbar.reset(n)
    pbar.set_description('Training')

    # enumerate through both dataloaders
    # the first dataloader contains the graph object and the second contains the weakly supervised labels
    for step, (batch, y_pred) in enumerate(zip(data_loader1, data_loader2)):
        batch = batch.to(device)
        batch_size = batch.batch.max().item()

        #get the fitness labels and zs predictors
        y = batch.y
        y = np.array(y, dtype=np.float32)
        y = torch.tensor(y, dtype=torch.float32)

        #may be a more efficient way to do this
        if model_config['weak_supervision'] == 1:
            y = torch.cat((y[:, 1:], y_pred[0].reshape(-1, 1)), 1).to(device)
        else:
            y = y[:, 1:].to(device)

        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        #neg_edge_index = batch.neg_edge_index.to(device)

        z = model.encode(data=batch)

        recon_loss = model.recon_loss(z, edge_index)
        total_recon_loss += recon_loss.item() * batch_size

        if model_config['kl_div_weight'] != 0:
            kl_div = model.kl_loss() #don't need to specify mu or loss since it will use the last one
            total_kl_div += kl_div.item() * batch_size
        else:
            kl_div = 0

        if model_config['zs_loss_weight'] != 0:
            zs_loss = model.zs_loss(z, edge_index, batch.batch, y)
            total_zs_loss += zs_loss.item() * batch_size
        else:
            zs_loss = 0

        loss = recon_loss + kl_div + zs_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.update()

    total_loss = total_recon_loss + total_kl_div + total_zs_loss

    avg_recon_loss = total_recon_loss / n
    avg_kl_div = total_kl_div / n
    avg_zs_loss = total_zs_loss / n

    return avg_recon_loss, avg_kl_div, avg_zs_loss

def eval(model_config: dict, model: nn.Module, device: torch.device, loader: DataLoader,
         pbar: tqdm) -> np.array:
    """Evaluates the model by extracting the embeddings.

    Args
    - model: nn.Module, GNN model, already moved to device
    - device: torch.device
    - loader: DataLoader
    - pbar: tqdm, progress bar

    Returns: np.array
    - extracting embeddings (before the decoding layer)
    """
    model.eval()
    save_embeddings = np.array([])

    pbar.reset(total=len(loader))
    pbar.set_description('Evaluating')

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            embedding = model.encode(extract=True, data=batch)
            embedding = embedding.cpu()
            embedding = embedding.reshape((-1, 56, model_config['hidden_dim']))

            #keep this if you want mean global pooling
            embedding = torch.mean(embedding, axis = 1)

            #keep this if you want all features from only the 4 mutated residues
            # a = embedding[:, 38:41,:]
            # b = embedding[:, 53, :].reshape(-1, 1, model_config['hidden_dim'])
            # embedding = np.concatenate((a, b), axis=1)
            # embedding = embedding.reshape(-1, embedding.shape[1]*embedding.shape[2])

            #old stuff
            # empty = np.zeros((embedding.shape[0], 32*4))
            # for i, row in enumerate(embedding):
            #     empty[i, :] = row.flatten()
            # embedding = empty

        #need to think about out how to get the most relevant features from the graph object
        if save_embeddings.shape[0] == 0:
            save_embeddings = embedding
        else:
            save_embeddings = np.concatenate([save_embeddings, embedding], axis=0)
        pbar.update()

    return save_embeddings


def train_supervised(N_train_samples = 384, n_splits = 5):
    fitness_df = pd.read_csv('/home/jyang4/repos/ProtGraphR/analysis/fitness.csv')
    dataset = GB1Dataset(dataframe = fitness_df, encoding = 'one-hot', attribute_names = [])
    dataset.encode_X()
    X_train_all = dataset.X
    y_train_all = fitness_df['fit'].values

    X_resample, y_resample = resample(X_train_all, y_train_all, n_samples=N_train_samples)
    kf = KFold(n_splits=n_splits)
    clfs = []
    y_pred_tests = np.zeros((n_splits, len(X_train_all)))

    for i, (train_index, test_index) in enumerate(kf.split(X_resample)):
        X_train, X_test = X_resample[train_index], X_resample[test_index]
        y_train, y_test = y_resample[train_index], y_resample[test_index]

        clf = Ridge(alpha=1)
        clf.fit(X_train, y_train)
        clfs.append(clf)

        y_pred_tests[i] = clf.predict(X_train_all)

    y_pred_test = np.mean(y_pred_tests, axis = 0)
    print(ndcg(y_train_all, y_pred_test))
    # with open('preds.npy', 'wb') as f:
    #     np.save(f, y_pred_test)
    return y_pred_test

def start_training(save_path: str, data_config: Mapping[str, Any],
                   model_config: Mapping[str, Any],
                   train_config: Mapping[str, Any],
                   device: str | torch.device) -> None:
    """
    Args:
        save_path: path to directory for saving outputs
        data_config: dict of dataset parameters
        model_config: dict of model parameters
        train_config: dict of training parameters
        device: str or torch.device
    """
    # Sample and fix a random seed if not set in train_config
    if 'seed' not in train_config:
        train_config['seed'] = random.randint(0, 9999)
    seed = train_config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Get model class
    model_class = get_model_class(model_config['name'])

    # Initialize dataset
    dataset, model_config = load_dataset(data_config, model_config)
    model = model_class(
        encoder=GraphEncoder(model_config=model_config),
        model_config=model_config,
        data_config=data_config).to(device)

    #Initialize dataloaders
    train_loader = DataLoader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=True)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = train_config['learning_rate'])

    #Get labels for weak supervision (does it even if no weak supervision as a filler)
    y_pred = train_supervised()
    dataset2 = TensorDataset(torch.tensor(y_pred, dtype= torch.float32))
    train_loader2 = torch.utils.data.DataLoader(dataset2, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=True)

    # Start training
    pbar = tqdm()
    for epoch in range(1, 1 + train_config['num_epochs']):
        recon_loss, kl_div, zs_loss = train(model_config, model, device, train_loader, train_loader2, optimizer, pbar)
        loss = recon_loss + kl_div + zs_loss

        #train_result = eval(model, device, train_loader, evaluator, pbar)
        #val_result = eval(model, device, valid_loader, evaluator, pbar)

        tqdm.write(f'Epoch {epoch:02d}, recon_loss: {recon_loss:.4f}, kl_div: {kl_div:.4f}, zs_loss: {zs_loss:.4f}')

        #update the best model after each epoch
        if epoch == 1 or loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path + '/best.pth')
            print('Best model saved')

def extract_features(save_path, data_config, model_config, train_config, device):

    # Get model class and load data
    model_class = get_model_class(model_config['name'])
    dataset, model_config = load_dataset(data_config, model_config, extract=True)

    #Use Pytorch's built in GAE/VGAE
    model = model_class(
        encoder=GraphEncoder(model_config=model_config),
        model_config=model_config,
        data_config=data_config).to(device)

    #Initialize dataloaders
    loader = DataLoader(dataset, batch_size=train_config['batch_size'], num_workers=train_config['num_workers'], shuffle=False)

    #Get best model
    model.load_state_dict(torch.load(save_path + '/best.pth'))
    pbar = tqdm()
    embeddings = eval(model_config, model, device, loader, pbar)

    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings)
    print("Saved Features: " + os.path.join(save_path, 'embeddings.npy'))

