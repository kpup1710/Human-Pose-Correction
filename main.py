import torch.nn.functional as F
import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import argparse
from data.dataset import TrainDataset, split_indices
import models as Model
from train_predictor import *
from train_corrector import *
from utils import *
import logging

if __name__ == "__main__":
  # Experiment options
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, default='workout')
  parser.add_argument('--data_path', type=str, default='dataset/workout_processed_data.json',
                        help='JSON file for data')
  parser.add_argument('--noised_data_path',type=str, default='dataset/noised_workout.json',
                        help='JSON file for noised data')
  parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
  parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
  parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')
  parser.add_argument('--epoch_pred', type=int, default=50, help='number of epoch(s) to train predictor')
  parser.add_argument('--epoch_cor', type=int, default=50, help='number of epoch(s) to train corrector')
  parser.add_argument('--lr_pred', type=float, default=4e-3, help='learning rate to train predictor')
  parser.add_argument('--lr_cor', type=float, default=3e-3, help='learning rate to train corrector')

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
  device = args.device
  checkpoints_path = f'checkpoints/' + args.name + '/'
  if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

  FORMAT = '%(asctime)s %(message)s'
  logging.basicConfig(format=FORMAT)
  d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}
# logging.basicConfig()
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  # logger.info('Protocol problem: %s', 'connection reset', extra=d)
  
  # logging
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
  edge = [ [1, 1, 2, 3, 5, 6, 1, 8, 9, 1, 11, 12, 1, 0, 14, 0, 15, 2, 5],  
    [2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 16, 15, 17, 16, 17],
    ]
  A = np.zeros((18,18))
  
  for i,j in zip(edge[0], edge[1]):
      A[i,j] = 1
      A[j, i] = 1
  

  device = get_default_device()
  logger.info(str(device) + " is available")

  model = Model.create_model(adj=torch.Tensor(A), opt=args)
  model = to_device(model, device)
  logger.info("Initialized model and put model to device")

  # Train predictor
  normal_dataset = TrainDataset(json_path=args.data_path, name=args.name)
  logger.info('Initialized normal dataset')

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
  
  train_dl_pred = DeviceDataLoader(train_dl_pred, device)
  val_dl_pred = DeviceDataLoader(val_dl_pred, device)

  num_epochs_pred = args.epoch_pred
  opt_fn = torch.optim.Adam
  lr_pred = args.lr_pred
  
  logger.info("Start training predictor")
  pred_train_losses, pred_val_losses, pred_val_metrics = his= train_predictor(num_epochs_pred, model.predictor, loss_func=F.cross_entropy,  train_dl=train_dl_pred, valid_dl=val_dl_pred, opt_fn=opt_fn, lr=lr_pred, metric=accuracy, PATH=checkpoints_path)
  
  
  # Train corrector
  model.load_predictor(path=checkpoints_path + 'best_predictor.pth')
  logger.info('Loaded predictor and set grad to False')
  noised_dataset = TrainDataset(json_path=args.noised_data_path, name='workout')
  logger.info("Initialized noised dataset")

  num_epochs_corr = args.epoch_cor
  lr_corr = args.lr_cor
  loss = yoga_loss()

  train_indices_cor, val_indices_cor, test_indices_cor = split_indices(len(noised_dataset), val_pct, rand_seed)
  
  batch_size = 128
  # Training sampler and data loader
  train_sampler_cor = SubsetRandomSampler(train_indices_cor)
  train_dl_cor = DataLoader(noised_dataset, batch_size, sampler=train_sampler_cor)
  
  # Validation set and data loader
  val_sampler_cor = SubsetRandomSampler(val_indices_cor)
  val_dl_cor = DataLoader(noised_dataset, batch_size, sampler=val_sampler_cor)
  
  test_sampler_cor = SubsetRandomSampler(test_indices_cor)
  test_dl_cor = DataLoader(noised_dataset, batch_size, sampler=test_sampler_cor)

  train_dl_cor = DeviceDataLoader(train_dl_cor, device)
  val_dl_cor = DeviceDataLoader(val_dl_cor, device)

  logger.info("Start training corrector")
  his= train_corrector(num_epochs_corr, model, loss_func=loss, train_dl=train_dl_cor, valid_dl=val_dl_cor, opt_fn=opt_fn, lr=lr_corr, metric=accuracy, PATH=checkpoints_path)