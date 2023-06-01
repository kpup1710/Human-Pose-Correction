import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse
from .data.dataset import TrainDataset, split_indices
import models as Model
from train_predictor import *
from train_corrector import *
from utils import *


if __name__ == "main":
  # Experiment options
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, default='workout')
  parser.add_argument('--data_path', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for data')
  parser.add_argument('--noised_data_path',type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for noised data')
  parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
  parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
  parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')

  # model options
    # predictor options
  parser.add_argument('--pred_hid_dim', type=int, default=128)
  parser.add_argument('--pred_coords_dim', type=tuple, default=(2, 128))
  parser.add_argument('--pred_num_layers', type=int, default=6)
  parser.add_argument('--pred_nodes_group', type=bool, default=None)
  parser.add_argument('--p_dropout', type=int, default=None)

    # corrector options
  parser.add_argument('--cor_dim_enc', type=int, default=128)
  parser.add_argument('--cor_depth', type=int, default=2)
  parser.add_argument('--cor_heads', type=int, default=2)
  parser.add_argument('--cor_dim_head', type=int, default=64)
  parser.add_argument('--cor_dropout', type=int, default=None)
  parser.add_argument('--cor_out_dim', type=int, default=2)

    # problem options
  parser.add_argument('--num_tokens', type=int, default=18, help='number of keypoints')
  parser.add_argument('--num_classes', type=int, default=22)

  # parse configs
  args = parser.parse_args()
  device = args['device']
  checkpoints_path = f'checkpoints/' + args['name'] + '/'
  normal_dataset = TrainDataset(json_path=args['data_path'], name=args['name'])
  
  edge = [ [1, 1, 2, 3, 5, 6, 1, 8, 9, 1, 11, 12, 1, 0, 14, 0, 15, 2, 5],  
    [2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 16, 15, 17, 16, 17],
    ]
  A = np.zeros((18,18))
  
  for i,j in zip(edge[0], edge[1]):
      A[i,j] = 1
      A[j, i] = 1
  
  
  # predictor = ModulatedGCN(adj=torch.tensor(A), hid_dim=128, coords_dim=(2, 128), num_layers=6, nodes_group=None, p_dropout=None, num_classes=22).double()
  model = Model.create_model(args)
  model = to_device(model, device)


  val_pct = 0.15
  rand_seed = 42
  train_indices, val_indices, test_indices = split_indices(len(normal_dataset), val_pct, rand_seed)
  
  batch_size = 128
  # Training sampler and data loader
  train_sampler = SubsetRandomSampler(train_indices)
  train_dl_pred = DataLoader(normal_dataset, batch_size, sampler=train_sampler)
  
  # Validation set and data loader
  val_sampler = SubsetRandomSampler(val_indices)
  val_dl_pred = DataLoader(normal_dataset, batch_size, sampler=val_sampler)
  
  test_sampler = SubsetRandomSampler(test_indices)
  test_dl_pred = DataLoader(normal_dataset, batch_size, sampler=test_sampler)
  
  # Train predictor
  num_epochs_pred = 50
  opt_fn = torch.optim.Adam
  lr_pred = 0.004
  
  pred_train_losses, pred_val_losses, pred_val_metrics = his= train_predictor(num_epochs_pred, model.predictor, loss_func=F.cross_entropy,  train_dl=train_dl_pred, valid_dl=val_dl_pred, opt_fn=opt_fn, lr=lr_pred, metric=accuracy, PATH=checkpoints_path)
  
  
  # Train corrector
  noised_dataset = TrainDataset(json_path=args['noised_data_path'])

  num_epochs_corr = 50
  lr_corr = 0.003
  loss = yoga_loss()

  train_dl = DataLoader(noised_dataset, batch_size, sampler=train_sampler)
  val_dl_pred = DataLoader(noised_dataset, batch_size, sampler=val_sampler)
  test_dl_pred = DataLoader(noised_dataset, batch_size, sampler=test_sampler)

  his= train_corrector(num_epochs_corr, model, loss_func=loss, train_dl=train_dl, valid_dl=val_dl, opt_fn=opt_fn, lr=lr_corr, metric=accuracy, PATH=checkpoints_path)