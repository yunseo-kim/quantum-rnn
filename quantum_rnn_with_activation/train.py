# Importing standard Qiskit libraries
from qiskit.utils import algorithm_globals

from qiskit.algorithms.optimizers import SPSA

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
# from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
backend = AerSimulator()

# Loading your IBM Quantum account(s)
# service = QiskitRuntimeService(channel="ibm_quantum")
# backend = service.backend("ibmq_qasm_simulator")

import torch
import os
import sys
from torch.utils.data import DataLoader
import numpy as np
from numpy.random import default_rng
import time
from trainer import Trainer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from classical_rnn.dataset import WeatherDataset
from model import pQRNN_RUS
import pickle

SEED = 23
np.random.seed(SEED)        # Seed for NumPy random number generator
rng = default_rng(SEED)
algorithm_globals.random_seed = SEED

# script_dir
script_dir = os.path.dirname(__file__)

# set device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
N_EPOCHS = 300
SEQUENCE_SIZE = 7
BATCH_SIZE = 4
N_SHOTS = 1000
N_QUBITS = 3  # Number of qubits allocated to each of two quantum registers. Total Number of Qubits = 2 * N_QUBITS
# Those parameters are determined by above parameters
N_ANSATZ_QUBITS = 2*N_QUBITS - 1
N_HIDDEN_QUBITS = N_QUBITS - 1
N_RUS_QUBITS = N_QUBITS - 1
N_TOTAL_QUBITS = N_ANSATZ_QUBITS + N_RUS_QUBITS
N_PARAMS = 8 * N_ANSATZ_QUBITS * SEQUENCE_SIZE
N_THETAS = N_RUS_QUBITS * SEQUENCE_SIZE

isReal = False
data_csv_path = os.path.join(script_dir, '../data/meteo_data.csv')

for lbl in ['max_temp', 'min_temp', 'avg_temp', 'max_wind_speed', 'avg_wind_speed', 'avg_pressure', 'avg_rel_humidity']:
  # train
  result_dir_path = os.path.join(script_dir, 'result/' + lbl + '/')
  
  # set model and optimizer
  model = pQRNN_RUS(backend=backend, isReal=isReal, n_shots=N_SHOTS, n_qubits=N_QUBITS, n_steps=SEQUENCE_SIZE)
  optimizer = SPSA(maxiter=10) #, snapshot_dir=result_dir_path+'snapshots/')

  # get dataset
  dataset_tr, dataset_val = WeatherDataset(SEQUENCE_SIZE, data_csv_path, lbl).getDataset()

  # set Dataloader for each dataset
  dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=BATCH_SIZE, shuffle=False)
  dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE)

  # scaler applied to input data
  scaler_path = os.path.join(script_dir, '../classical_rnn/result/'+ lbl + '/scaler.pkl')
  with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

  initial_params = algorithm_globals.random.random(N_PARAMS)
  initial_thetas = algorithm_globals.random.random(N_HIDDEN_QUBITS)
  trainer = Trainer(model=model, optimizer=optimizer, result_dir_path=result_dir_path)

  start = time.time()
  for epoch_id in range(N_EPOCHS):
    optimized_params = trainer.train(initial_params, initial_thetas, dataloader_tr, epoch_id)
    trainer.validate(optimized_params, dataloader_val, epoch_id, scaler)
    # save best result
    if trainer.score == np.min(trainer.score_lst):
      torch.save(optimized_params, result_dir_path+'/best.pt')
      # torch.save(optimizer.state_dict(), result_dir_path+'/best_optim.pt')

  elapsed = time.time() - start
  print(f"Fit in {elapsed:0.2f} seconds")