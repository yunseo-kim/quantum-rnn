# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.utils import algorithm_globals

# Importing Numpy
import numpy as np
from numpy.random import default_rng

# from IPython.display import clear_output


# SAVE_PATH = "aeqis/" # Data saving folder
SEED = 23
np.random.seed(SEED)        # Seed for NumPy random number generator
rng = default_rng(SEED)
algorithm_globals.random_seed = SEED


class pQRNN_RUS():
  """
  Staggered Quantum RNN
  """
  def __init__(self, backend, isReal: bool, n_shots: int, n_qubits: int, n_steps: int):
    super(pQRNN_RUS, self).__init__()
    self.backend = backend
    self.isReal = isReal
    self.n_shots = n_shots
    self.n_qubits = n_qubits
    self.n_steps = n_steps
    self.n_params = 8 * n_qubits * n_steps
    
    # Setup a base quantum circuit for our experiments
    """
    reg_d is used to embed the sequential data, one element at each time step
    reg_h is used to store information about the history of all previous elements
    """
    self.qr = QuantumRegister(2*self.n_qubits)
    self.reg_d = self.qr[0:self.n_qubits]
    self.reg_h = self.qr[self.n_qubits:2*self.n_qubits]

    # Output at the final time step
    self.cr = ClassicalRegister(1)
    self.output_bit = self.cr[0]

    # Adjustable parameters to be optimized in the learning process
    self.params = ParameterVector('P', self.n_params)
    self.rus_angles = ParameterVector('RUS', self.n_params)
    self.input_seq = ParameterVector('x', self.n_steps)

    # Initializing
    def initialize_circuit() -> QuantumCircuit:
        return QuantumCircuit(self.qr, self.cr)
    
    self.initial_state = initialize_circuit()

    def encode_angle(qc: QuantumCircuit, input_data, qr: QuantumRegister, n_qubits: int) -> QuantumCircuit:
        # input data range: (0,1)
        input_data = input_data * 2 - 1  # rescaled range: (-1,1)
        encoded_angle = np.arccos(input_data)  # encoded angle range: (0,PI)
        for i in range(n_qubits):
            qc.ry(encoded_angle, qr[i])
        return qc
    
    def apply_single_qubit_gates(qc: QuantumCircuit, qr: QuantumRegister, params, n_qubits: int) -> QuantumCircuit:
        if not self.isReal:
            qc.barrier()
        for i in range(2*n_qubits):
            qc.rx(params[3*i], qr[i])
            qc.rz(params[3*i+1], qr[i])
            qc.rx(params[3*i+2], qr[i])
        return qc
    
    def apply_two_qubit_gates(qc: QuantumCircuit, qr: QuantumRegister, params, n_qubits: int) -> QuantumCircuit:
        if not self.isReal:
            qc.barrier()
        for i in range(-2*n_qubits, 0):
            qc.cx(qr[i], qr[i+1])
            qc.rz(params[i], qr[i+1])
            qc.cx(qr[i], qr[i+1])
        return qc
    
    def apply_ansatz(qc: QuantumCircuit, qr: QuantumRegister, params, n_qubits: int) -> QuantumCircuit:
        qc = apply_single_qubit_gates(qc, qr, params[0:6*n_qubits], n_qubits)
        qc = apply_two_qubit_gates(qc, qr, params[6*n_qubits:8*n_qubits], n_qubits)
        return qc
    
    def apply_stagger(reg_d: list, reg_h: list, n_qubits: int) -> tuple[list, list]:
        tmp = reg_d[0]
        for i in range(n_qubits-1):
            reg_d[i] = reg_d[i+1]
        reg_d[n_qubits-1] = reg_h[0]
    
        for i in range(n_qubits-1):
            reg_h[i] = reg_h[i+1]
        reg_h[n_qubits-1] = tmp
    
        return reg_d, reg_h

    def apply_partial_measurement(qc: QuantumCircuit, qr: QuantumRegister, c_bit) -> QuantumCircuit:
        qc.measure(qr[0], c_bit)
        return qc

    def apply_qrb(qc: QuantumCircuit, reg_d: list, reg_h: list, params, n_qubits: int, timestep_idx: int) -> QuantumCircuit:
        if timestep_idx > 0:
            if not self.isReal:
                qc.barrier()
            qc.reset(reg_d)
        qc = encode_angle(qc, self.input_seq[timestep_idx], reg_d, n_qubits)
        qc = apply_ansatz(qc, self.qr, params[timestep_idx*8*n_qubits:(timestep_idx+1)*8*n_qubits], n_qubits)
        (reg_d, reg_h) = apply_stagger(reg_d, reg_h, n_qubits)
        # qc = apply_partial_measurement(qc, reg_d, cr)
        return qc
    
    pqrnn_RUS: QuantumCircuit = initialize_circuit()
    for step in range(self.n_steps):
        pqrnn_RUS = apply_qrb(pqrnn_RUS, self.reg_d, self.reg_h, self.params, self.n_qubits, step)
    self.pqrnn_RUS = apply_partial_measurement(pqrnn_RUS, self.reg_d, self.output_bit)

    """
    initial_params = algorithm_globals.random.random(self.n_params)
    self.sqrnn = sqrnn.assign_parameters(initial_params, inplace=False)
    """


  def forward(self, input_batch: np.ndarray, params_values: np.ndarray) -> np.ndarray:
    batch_size = len(input_batch)
    y: np.ndarray = np.zeros((batch_size,1))

    for batch_idx in range(batch_size):
        x = input_batch[batch_idx]
        pqrnn_RUS = self.pqrnn_RUS.assign_parameters({self.input_seq[i]:x[i] for i in range(self.n_steps)}, inplace=False)
        pqrnn_RUS.assign_parameters(params_values, inplace=True)

        if self.isReal:
            pqrnn_RUS = transpile(pqrnn_RUS, self.backend) #, initial_layout=initial_layout)
    
         # We run the simulation and get the counts
        counts: dict = self.backend.run(pqrnn_RUS, shots=self.n_shots).result().get_counts()
        if '1' in counts:
            result: float = counts['1'] / self.n_shots
        else:
            result = 0

        y[batch_idx][0] = result

    return y