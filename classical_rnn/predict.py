import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from model import ClassicRNN
import matplotlib.pyplot as plt
import pickle

# script_dir
script_dir = os.path.dirname(__file__)

# set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
n_epochs = 500
sequence_size = 7
input_size = 1
hidden_size = 6
num_layers = 5
data_csv_path = os.path.join(script_dir, '../data/meteo_data.csv')

# get datetime object



for lbl in ['max_temp', 'min_temp', 'avg_temp', 'max_wind_speed', 'avg_wind_speed', 'avg_pressure', 'avg_rel_humidity']:
  # set model and optimizer
  model = ClassicRNN(device, input_size, hidden_size, num_layers, dropout_ratio=0.2)
  optimizer = optim.Adam(model.parameters())

  # get model
  model.load_state_dict(torch.load(os.path.join(script_dir, './result/' + lbl + '/best.pt')))
  
  # get dataset
  weather_dataset = WeatherDataset(sequence_size, data_csv_path, lbl)
  dataset_tr, dataset_val = weather_dataset.getDataset()

  # set Dataloader for each dataset
  dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=1, shuffle=False)
  dataloader_val = DataLoader(dataset=dataset_val, batch_size=1)

  # load model to gpu
  model = model.to(device)

  # make result dir
  result_dir_path = os.path.join(script_dir, './result/' + lbl + '/')
  
  # scaler applied to input data
  with open(result_dir_path+'scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

  model.eval()
  
  # plot pred data
  date_list = weather_dataset.getDateData(idx_st=sequence_size)
  pred_list = []
  pred_list_val = []
  actual_list = weather_dataset.getActualData(idx_st=sequence_size)
  
  for x, y in dataloader_tr:
    x = x.to(device)
    y = y.to(device)
    y_pred = model.forward(x)
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = scaler.inverse_transform(y_pred)
    pred_list.append(y_pred[0][0])

  for x, y in dataloader_val:
    x = x.to(device)
    y = y.to(device)
    y_pred = model.forward(x)
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = scaler.inverse_transform(y_pred)
    pred_list.append(y_pred[0][0])
    pred_list_val.append(y_pred[0][0])

  plt.figure()
  plt.plot(date_list, actual_list, '--', label="actual")
  plt.plot(date_list, pred_list, label="train+val pred")
  plt.legend()
  plt.xticks(rotation=45)
  plt.title('Classic RNN Prediction (train+val)')
  plt.savefig(os.path.join(result_dir_path, 'pred_graph_all.png'))
  plt.close()

  train_size = len(pred_list)-len(pred_list_val)
  plt.figure()
  plt.plot(date_list[train_size:], actual_list[train_size:], '--', label="actual")
  plt.plot(date_list[train_size:], pred_list_val, label="val pred")
  plt.legend()
  plt.xticks(rotation=45)
  plt.title('Classic RNN Prediction (train+val)')
  plt.savefig(os.path.join(result_dir_path, 'pred_graph_val.png'))
  plt.close()