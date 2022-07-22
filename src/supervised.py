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

