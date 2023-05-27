import pandas as pd
import torchvision.models as models
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import json
import time
from tqdm import tqdm
from .data.dataset import TrainDataset, split_indices
from .model import Predictor_Corrector
from train_predictor import *
from train_corrector import *
from utils import *
from .model.Predictor import ModulatedGCN

device = get_default_device()
dataset = TrainDataset(json_path='/content/workout_processed_data.json')

edge = [ [1, 1, 2, 3, 5, 6, 1, 8, 9, 1, 11, 12, 1, 0, 14, 0, 15, 2, 5],  
  [2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 16, 15, 17, 16, 17],
  ]
A = np.zeros((18,18))

for i,j in zip(edge[0], edge[1]):
    A[i,j] = 1
    A[j, i] = 1

predictor = ModulatedGCN(adj=torch.tensor(A), hid_dim=128, coords_dim=(2, 128), num_layers=6, nodes_group=None, p_dropout=None, num_classes=22).double()

val_pct = 0.15
rand_seed = 42
train_indices, val_indices, test_indices = split_indices(len(dataset), val_pct, rand_seed)

batch_size = 128
# Training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset, batch_size, sampler=train_sampler)

# Validation set and data loader
val_sampler = SubsetRandomSampler(val_indices)
val_dl = DataLoader(dataset, batch_size, sampler=val_sampler)

test_sampler = SubsetRandomSampler(test_indices)
test_dl = DataLoader(dataset, batch_size, sampler=test_sampler)

# Train predictor
num_epochs_pred = 50
opt_fn = torch.optim.Adam
lr_pred = 0.004

pred_train_losses, pred_val_losses, pred_val_metrics = his= train_predictor(num_epochs_pred, predictor, loss_func=F.cross_entropy, train_dl=train_dl, valid_dl=val_dl, opt_fn=opt_fn, lr=lr_pred, metric=accuracy)


# Train corrector
model = Predictor_Corrector(adj=torch.tensor(A), hid_dim=128, coords_dim=(2, 128), num_layers_pred=6, nodes_group=None, p_dropout=None, num_classes=22,
                            num_tokens=18, depth=2, heads=2, dim_head=64).double().cuda()

num_epochs_corr = 50
lr_corr = 0.003
loss = yoga_loss()
model = to_device(model, device)

his= train_corrector(num_epochs_corr, model, loss_func=loss, train_dl=train_dl, valid_dl=val_dl, opt_fn=opt_fn, lr=lr_corr, metric=accuracy)