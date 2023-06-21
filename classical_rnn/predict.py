import torch
import torch.optim as optim
import os
import numpy as np
from trainer import Trainer
from dataset import WeatherDataset
from model import ClassicRNN
import matplotlib.pyplot as plt

# script_dir
script_dir = os.path.dirname(__file__)

# set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
n_epochs = 500
sequence_size = 7
input_size = 1
hidden_size = 4
num_layers = 2
tr_csv_path = os.path.join(script_dir, '../data/meteo_data_tr.csv')
val_csv_path = os.path.join(script_dir, '../data/meteo_data_val.csv')

# get datetime object



for lbl in ['max_temp', 'min_temp', 'avg_temp', 'max_wind_speed', 'avg_wind_speed', 'avg_pressure', 'avg_rel_humidity']:
  # set model and optimizer
  model = ClassicRNN(device, input_size, hidden_size, num_layers, dropout_ratio=0.2)
  optimizer = optim.Adam(model.parameters())

  # get model
  model.load_state_dict(torch.load(os.path.join(script_dir, './result/' + lbl + '/best.pt')))
  
  # get dataset
  weather_dataset = WeatherDataset(sequence_size, tr_csv_path, val_csv_path, lbl)
  _, dataset_val = weather_dataset.getDataset()

  # load model to gpu
  model = model.to(device)

  # make result dir
  result_dir_path = os.path.join(script_dir, './result/' + lbl + '/')

  trainer = Trainer(device=device, model=model, optimizer=optimizer, result_dir_path=result_dir_path)

  x, _ = dataset_val[0]
  x = x.to(device)
  x = x.unsqueeze(-1).transpose(0, 1)
  
  model.eval()
  
  pred_list = []

  for i in range(60):
    # validate model
    y = model.forward(x).squeeze(-1)
    x = x.squeeze()
    # print(x.shape, y.shape)
    x = torch.cat((x[1:], y)).unsqueeze(0)
    pred_list.append(y.item())
  
  plt.figure()
  plt.plot(weather_dataset.val_date_data[7:67], pred_list, label="pred")
  plt.plot(weather_dataset.val_date_data[7:67], dataset_val.get_real_data(7, 67), label="real")
  plt.legend()
  plt.title('Classic RNN Prediction')
  plt.savefig(os.path.join(result_dir_path, 'pred_graph.png'))
  plt.close()