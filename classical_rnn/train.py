import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import numpy as np
from trainer import Trainer
from dataset import WeatherDataset
from model import ClassicRNN

# script_dir
script_dir = os.path.dirname(__file__)

# set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
n_epochs = 100
sequence_size = 7
input_size = 1
hidden_size = 4
num_layers = 2
batch_size = 1
tr_csv_path = os.path.join(script_dir, '../data/meteo_data_tr.csv')
val_csv_path = os.path.join(script_dir, '../data/meteo_data_val.csv')

for lbl in ['max_temp', 'min_temp', 'avg_temp', 'max_wind_speed', 'avg_wind_speed', 'avg_pressure', 'avg_rel_humidity']:
  # set model and optimizer
  model = ClassicRNN(device, input_size, hidden_size, num_layers, dropout_ratio=0.2)
  optimizer = optim.Adam(model.parameters())

  # get dataset
  dataset_tr, dataset_val = WeatherDataset(sequence_size, tr_csv_path, val_csv_path, lbl).getDataset()

  # set Dataloader for each dataset
  dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=False)
  dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size)

  # load model to gpu
  model = model.to(device)

  # make result dir
  result_dir_path = os.path.join(script_dir, './result/' + lbl + '/')

  trainer = Trainer(device=device, model=model, optimizer=optimizer, result_dir_path=result_dir_path)

  for epoch_id in range(n_epochs):
    trainer.train(dataloader_tr, epoch_id)
    trainer.validate(dataloader_val, epoch_id)
    # save best result
    if trainer.score == np.min(trainer.score_lst):
      torch.save(model.state_dict(), result_dir_path+'/best.pt')
      torch.save(optimizer.state_dict(), result_dir_path+'/best_optim.pt')