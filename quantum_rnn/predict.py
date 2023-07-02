import torch
import os, sys
from torch.utils.data import DataLoader
from model import sQRNN
import matplotlib.pyplot as plt
import pickle
from qiskit_aer import AerSimulator
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from classical_rnn.dataset import WeatherDataset
from tqdm import tqdm
import pandas as pd

# script_dir
script_dir = os.path.dirname(__file__)

# set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# backend
backend = AerSimulator()

# parameters
N_EPOCHS = 300
SEQUENCE_SIZE = 7
BATCH_SIZE = 1
N_SHOTS = 500
N_QUBITS = 3  # Number of qubits allocated to each of two quantum registers. Total Number of Qubits = 2 * N_QUBITS
N_PARAMS = 8 * N_QUBITS * SEQUENCE_SIZE
isReal = False
data_csv_path = os.path.join(script_dir, '../data/meteo_data.csv')


for lbl in ['max_temp', 'avg_wind_speed', 'max_wind_speed', 'min_temp', 'avg_temp', 'avg_pressure', 'avg_rel_humidity']:
  # set model and optimizer
  model = sQRNN(backend=backend, isReal=isReal, n_shots=N_SHOTS, n_qubits=N_QUBITS, n_steps=SEQUENCE_SIZE)
  
  # get dataset
  weather_dataset = WeatherDataset(SEQUENCE_SIZE, data_csv_path, lbl)
  dataset_tr, dataset_val = weather_dataset.getDataset()

  # set Dataloader for each dataset
  dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=BATCH_SIZE, shuffle=False)
  dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE)

  # make result dir
  result_dir_path = os.path.join(script_dir, './result/' + lbl + '/')
  
  # scaler applied to input data
  scaler_path = os.path.join(script_dir, '../classical_rnn/result/'+ lbl + '/scaler.pkl')
  with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

  # get model
  optim_params = torch.load(os.path.join(script_dir, './result/' + lbl + '/best.pt'))

  # plot pred data
  date_list = weather_dataset.getDateData(idx_st=SEQUENCE_SIZE)
  pred_list = []
  pred_list_val = []
  actual_list = weather_dataset.getActualData(idx_st=SEQUENCE_SIZE)
  
  for x, y in tqdm(dataloader_tr):
    x, y = x.detach().numpy(), y.detach().numpy()
    y_pred = model.forward(x, optim_params)
    y_pred = y_pred
    y_pred = scaler.inverse_transform(y_pred)
    pred_list.append(y_pred[0][0])

  for x, y in tqdm(dataloader_val):
    x, y = x.detach().numpy(), y.detach().numpy()
    y_pred = model.forward(x, optim_params)
    y_pred = y_pred
    y_pred = scaler.inverse_transform(y_pred)
    pred_list.append(y_pred[0][0])
    pred_list_val.append(y_pred[0][0])

  plt.figure()
  plt.plot(date_list, actual_list, '--', label="actual")
  plt.plot(date_list, pred_list, label="train+val pred")
  plt.legend()
  plt.xticks(rotation=45)
  plt.title('Quantum RNN Prediction (train+val)')
  plt.savefig(os.path.join(result_dir_path, 'pred_graph_all.png'))
  plt.close()

  train_size = len(pred_list)-len(pred_list_val)
  plt.figure()
  plt.plot(date_list[train_size:train_size+10], actual_list[train_size:train_size+10], '--', label="actual")
  plt.plot(date_list[train_size:train_size+10], pred_list_val[:10], label="val pred")
  plt.legend()
  plt.xticks(rotation=45)
  plt.title('Quantum RNN Prediction (train+val)')
  plt.savefig(os.path.join(result_dir_path, 'pred_graph_val.png'))
  plt.close()

  pred_data = {'date':[date.strftime('%Y/%m/%d') for date in date_list], lbl+'_pred':pred_list, lbl:actual_list}
  df = pd.DataFrame(data=pred_data)
  df.to_csv(os.path.join(result_dir_path, 'pred_result.csv'))