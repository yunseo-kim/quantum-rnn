import torch
from torch.utils.data import Dataset, TensorDataset
import pandas as pd
from datetime import datetime
from typing import List
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import numpy as np

script_dir = os.path.dirname(__file__)

class WeatherDataset(Dataset):
  """
  Get Weather Dataset
  """
  def __init__(self, sequence_size, data_csv_path, label, val_ratio=0.4):
    self.scaler = MinMaxScaler()
    self._dataset = CsvDataset(data_csv_path, label, sequence_size, self.scaler)
    train_size = int(len(self._dataset)*(1-val_ratio))
    self.tr_dataset = SplittedDataset([self._dataset[i] for i in range(train_size)])
    self.val_dataset = SplittedDataset([self._dataset[i] for i in range(train_size, len(self._dataset))])
    date_data = pd.read_csv(data_csv_path, dtype=str)['date'].values.tolist()
    self._date_data = [datetime.strptime(date, '%Y/%m/%d') for date in date_data]
    
    with open(os.path.join(script_dir, './result/'+label+'/scaler.pkl'), 'wb') as f:
      pickle.dump(self.scaler, f)
    f.close

  def getDataset(self):
    return self.tr_dataset, self.val_dataset

  def getActualData(self, idx_st=None, idx_ed=None):
    if idx_st != None and idx_ed != None:
      data = self.scaler.inverse_transform(np.expand_dims(self._dataset.data[idx_st:idx_ed], axis=-1))
      return np.squeeze(data)
    elif idx_st == None and idx_ed != None:
      data = self.scaler.inverse_transform(np.expand_dims(self._dataset.data[:idx_ed], axis=-1))
      return np.squeeze(data)
    elif idx_st != None and idx_ed == None:
      data = self.scaler.inverse_transform(np.expand_dims(self._dataset.data[idx_st:], axis=-1))
      return np.squeeze(data)
    else:
      data = self.scaler.inverse_transform(np.expand_dims(self._dataset.data, axis=-1))
      return np.squeeze(data)

  def getDateData(self, idx_st=None, idx_ed=None):
    if idx_st != None and idx_ed != None:
      return self._date_data[idx_st:idx_ed]
    elif idx_st == None and idx_ed != None:
      return self._date_data[:idx_ed]
    elif idx_st != None and idx_ed == None:
      return self._date_data[idx_st:]
    else:
      return self._date_data

class CsvDataset(Dataset):
  """
  Get Csv Dataset
  """
  def __init__(self, csv_path, label, sequence_size, scaler):
    data = np.array(pd.read_csv(csv_path)[label].values.tolist())
    data = np.expand_dims(data, axis=-1)
    self.data = np.squeeze(scaler.fit_transform(data))
    self.sequence_size = sequence_size

  def __getitem__(self, index):
    x = torch.FloatTensor(self.data[index:index+self.sequence_size])
    y = torch.FloatTensor([self.data[index+self.sequence_size]])
    return x, y
  
  def __len__(self):
    return len(self.data)-self.sequence_size

class SplittedDataset(Dataset):
  def __init__(self, splitted_dataset:List):
    self._data = splitted_dataset

  def __getitem__(self, index):
    x, y = self._data[index]
    return x, y
  
  def __len__(self):
    return len(self._data)