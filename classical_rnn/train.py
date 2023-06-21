import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import numpy as np
from trainer import Trainer
from dataset import WeatherDataset
from model import ClassicRNN
import pickle

# script_dir
script_dir = os.path.dirname(__file__)

# set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
n_epochs = 300
sequence_size = 7
input_size = 1
hidden_size = 6
num_layers = 5
batch_size = 4
data_csv_path = os.path.join(script_dir, '../data/meteo_data.csv')

for lbl in ['max_temp', 'min_temp', 'avg_temp', 'max_wind_speed', 'avg_wind_speed', 'avg_pressure', 'avg_rel_humidity']:
  # set model and optimizer
  model = ClassicRNN(device, input_size, hidden_size, num_layers, dropout_ratio=0.2)
  optimizer = optim.Adam(model.parameters())

  # get dataset
  dataset_tr, dataset_val = WeatherDataset(sequence_size, data_csv_path, lbl).getDataset()

  # set Dataloader for each dataset
  dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=False)
  dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size)

  # load model to gpu
  model = model.to(device)

  # train
  result_dir_path = os.path.join(script_dir, './result/' + lbl + '/')

  # scaler applied to input data
  with open(result_dir_path+'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

  trainer = Trainer(device=device, model=model, optimizer=optimizer, result_dir_path=result_dir_path)

  for epoch_id in range(n_epochs):
    trainer.train(dataloader_tr, epoch_id)
    trainer.validate(dataloader_val, epoch_id, scaler)
    # save best result
    if trainer.score == np.min(trainer.score_lst):
      torch.save(model.state_dict(), result_dir_path+'/best.pt')
      torch.save(optimizer.state_dict(), result_dir_path+'/best_optim.pt')