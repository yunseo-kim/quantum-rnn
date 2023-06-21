import torch
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime

class WeatherDataset(Dataset):
  """
  Get Weather Dataset
  """
  def __init__(self, sequence_size, tr_csv_path, val_csv_path, label):
    self.tr_dataset = CsvDataset(tr_csv_path, label, sequence_size)
    self.val_dataset = CsvDataset(val_csv_path, label, sequence_size)
    self.tr_date_data = pd.to_datetime(pd.read_csv(tr_csv_path)['date']).values.tolist()
    self.val_date_data = pd.to_datetime(pd.read_csv(val_csv_path)["date"]).values.tolist()
    
    # self.tr_date_data = [datetime.strftime(date_str, '%Y/%m/%d') for date_str in tr_date_data]
    # self.val_date_data = [datetime.strftime(date_str, '%Y/%m/%d') for date_str in val_date_data]

  def getDataset(self):
    return self.tr_dataset, self.val_dataset

class CsvDataset(Dataset):
  """
  Get Csv Dataset
  """
  def __init__(self, csv_path, label, sequence_size):
    self.data = pd.read_csv(csv_path)[label].values.tolist()
    self.sequence_size = sequence_size

  def __getitem__(self, index):
    x = torch.FloatTensor(self.data[index:index+self.sequence_size])
    y = torch.FloatTensor([self.data[index+self.sequence_size]])
    return x, y
  
  def __len__(self):
    return len(self.data)-self.sequence_size
  
  def get_real_data(self, idx_st, idx_ed):
    return self.data[idx_st:idx_ed]