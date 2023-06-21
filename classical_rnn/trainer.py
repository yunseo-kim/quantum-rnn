import torch.nn as nn
from tqdm import tqdm
from torch.nn.functional import mse_loss
import numpy as np
import matplotlib.pyplot as plt

class Trainer:
  def __init__(self, device, model, optimizer, result_dir_path):
    self.device = device
    self.model:nn.Module = model
    self.optimizer = optimizer
    self.loss = 0
    self.score = 0
    self.loss_lst_tr = []
    self.loss_lst_val = []
    self.score_lst = []
    self.result_path = result_dir_path

  def train(self, dataloader_tr, epoch_idx):
    self.model.train()
    self.loss = 0
    for x, y in tqdm(dataloader_tr):
      x = x.to(self.device)
      y = y.to(self.device)
      
      y_pred = self.model.forward(x)
      loss = mse_loss(y_pred, y)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      self.loss += loss.item()

    self.loss = self.loss / len(dataloader_tr)
    print(f"Train Epoch {epoch_idx} | MSE_loss : {self.loss}")
    self.loss_lst_tr.append(self.loss)

  def validate(self, dataloader_val, epoch_idx):
    # validate model
    self.model.eval()
    self.loss = 0
    for x, y in tqdm(dataloader_val):
      x = x.to(self.device)
      y = y.to(self.device)
      y_pred = self.model.forward(x)
      loss = mse_loss(y_pred, y)
      self.loss += loss.item()

    self.loss = self.loss / len(dataloader_val)
    self.score = np.sqrt(self.loss)
    self.score = np.sqrt(self.score / len(dataloader_val))
    print(f"Validation Epoch {epoch_idx} | MSE_loss : {self.loss}, RMSE_loss : {self.score}")
    
    self.loss_lst_val.append(self.loss)
    self.score_lst.append(self.score)

    # make graph
    self.make_loss_graph(self.loss_lst_tr, self.loss_lst_val, self.result_path)
    self.make_score_graph(self.score_lst, self.result_path)

  def make_loss_graph(self, loss_lst_tr, loss_lst_val, dir_path):
      plt.figure()
      x = list(range(0, len(loss_lst_tr)))
      plt.plot(x, loss_lst_tr, label="train")
      x = list(range(0, len(loss_lst_val)))
      plt.plot(x, loss_lst_val, label="val")
      plt.yscale('log')
      plt.legend()
      plt.title('loss of each epoch')
      plt.savefig(dir_path+'/loss.png')
      plt.close()

  def make_score_graph(self, score_lst_tr, dir_path):
      plt.figure()
      x = list(range(0, len(score_lst_tr)))
      plt.plot(x, score_lst_tr)
      plt.title('score of each epoch')
      plt.savefig(dir_path+'/score.png')
      plt.close()