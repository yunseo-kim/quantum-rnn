# Importing standard Qiskit libraries
from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals

#from qiskit_machine_learning.utils.loss_functions import L2Loss

from tqdm import tqdm
import torch
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import time

SEED = 23
np.random.seed(SEED)        # Seed for NumPy random number generator
rng = default_rng(SEED)
algorithm_globals.random_seed = SEED

class Trainer:
  def __init__(self, model, optimizer, result_dir_path: str):
    self.model = model
    self.optimizer = optimizer
    self.loss = 0.0
    self.score = 0.0
    self.loss_lst_tr: list = []
    self.loss_lst_val: list = []
    self.score_lst: list = []
    self.result_path = result_dir_path

  def train(self, params_values: np.ndarray, dataloader_tr, epoch_idx: int) -> np.ndarray:
    # train model
    self.loss = 0
    opt_params = params_values
    
    for x, y in tqdm(dataloader_tr):
      start = time.time()
      x = x.numpy()
      y = y.numpy()

      def loss_func(params_values):
        y_pred: np.ndarray = self.model.forward(x, params_values)
        #loss = L2Loss.evaluate(y_pred, y)
        loss = np.linalg.norm(y_pred - y)
        return loss

      opt_result = self.optimizer.minimize(loss_func, opt_params)
      opt_params: np.ndarray = opt_result.x
      self.loss += (loss_func(opt_params))**2

      elapsed = time.time() - start
      print(f"Batch 실행 시간: {elapsed:0.2f} seconds")
      print(f"loss = {(opt_result.fun)**2}")

    self.loss = np.sqrt(self.loss / len(dataloader_tr))
    print(f"Train Epoch {epoch_idx} | MSE_loss : {self.loss}")
    self.loss_lst_tr.append(self.loss)
    return opt_params

  def validate(self, params_values: np.ndarray, dataloader_val, epoch_idx: int, scaler):
    # validate model
    self.loss = 0

    for x, y in tqdm(dataloader_val):
      x = x.numpy()
      y = y.numpy()
      y_pred: np.ndarray = self.model.forward(x, params_values)
      self.loss += (np.linalg.norm(y_pred - y))**2

      # compute score
      y_pred_: np.ndarray = scaler.inverse_transform(y_pred)
      y_: np.ndarray = scaler.inverse_transform(y)
      self.score += (np.linalg.norm(y_pred_ - y_))**2

    self.loss = self.loss / len(dataloader_val)
    self.score = np.sqrt(self.score / len(dataloader_val))
    
    print(f"Validation Epoch {epoch_idx} | MSE_loss : {self.loss}, RMSE_loss : {self.score}")
    
    self.loss_lst_val.append(self.loss)
    self.score_lst.append(self.score)

    # make graph
    self.make_loss_graph(self.loss_lst_tr, self.loss_lst_val, self.result_path)
    self.make_score_graph(self.score_lst, self.result_path)

  def make_loss_graph(self, loss_lst_tr: list, loss_lst_val: list, dir_path: str):
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

  def make_score_graph(self, score_lst_tr: list, dir_path: str):
      plt.figure()
      x = list(range(0, len(score_lst_tr)))
      plt.plot(x, score_lst_tr)
      plt.title('score of each epoch')
      plt.savefig(dir_path+'/score.png')
      plt.close()