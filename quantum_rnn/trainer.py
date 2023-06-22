# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister, Aer
from qiskit.utils import algorithm_globals

#from qiskit_machine_learning.neural_networks import EstimatorQNN
#from qiskit.primitives import Estimator
from qiskit_machine_learning.utils.loss_functions import L2Loss

from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import time

SEED = 23
np.random.seed(SEED)        # Seed for NumPy random number generator
rng = default_rng(SEED)
algorithm_globals.random_seed = SEED

class Trainer:
  def __init__(self, model: QuantumCircuit.quantumcircuit.QuantumCircuit, params_values: np.ndarray, optimizer, result_dir_path):
    self.model = model
    self.params_values = params_values
    self.optimizer = optimizer
    self.loss = 0
    self.score = 0
    self.loss_lst_tr = []
    self.loss_lst_val = []
    self.score_lst = []
    self.result_path = result_dir_path

  def train(self, dataloader_tr, epoch_idx) -> np.ndarray:
    # train model
    self.loss = 0
    opt_params = self.params_values
    
    for x, y in tqdm(dataloader_tr):
      x = x.numpy()
      y = y.numpy()
      
      def loss_func(params_values: np.ndarray) -> float:
        y_pred = self.model.forward(x, params_values)
        loss = L2Loss(y_pred, y)
        return loss

      start = time.time()
      opt_params = self.optimizer.minimize(loss_func, opt_params)
      elapsed = time.time() - start

      self.loss += loss_func(opt_params)

    self.loss = self.loss / len(dataloader_tr)
    print(f"Train Epoch {epoch_idx} | MSE_loss : {self.loss}")
    self.loss_lst_tr.append(self.loss)
    return opt_params

  def validate(self, dataloader_val, epoch_idx, scaler):
    # validate model
    self.loss = 0
    for x, y in tqdm(dataloader_val):
      x = x.numpy()
      y = y.numpy()
      y_pred = self.model.forward(x, self.params_values)
      self.loss += L2Loss(y_pred, y)

      # compute score
      y_pred_ = scaler.inverse_transform(y_pred)
      y_ = scaler.inverse_transform(y)
      self.score += L2Loss(y_pred_, y_)

    self.loss = self.loss / len(dataloader_val)
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