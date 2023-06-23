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

    self.N_ANSATZ_QUBITS = 2*self.n_qubits - 1
    self.N_HIDDEN_QUBITS = self.n_qubits - 1
    self.N_RUS_QUBITS = self.n_qubits - 1
    self.N_TOTAL_QUBITS = self.N_ANSATZ_QUBITS + self.N_RUS_QUBITS
    self.N_PARAMS = 8 * self.N_ANSATZ_QUBITS * self.n_steps
    self.N_THETAS = self.N_RUS_QUBITS * self.n_steps
    
    # Setup a base quantum circuit for our experiments
    """
    reg_d is used to embed the sequential data, one element at each time step
    reg_h is used to store information about the history of all previous elements
    """
    qr_ansatz = QuantumRegister(self.N_ANSATZ_QUBITS + self.N_ANSATZ_QUBITS)
    reg_d = qr_ansatz[0:self.n_qubits]
    reg_h = qr_ansatz[self.n_qubits:self.N_ANSATZ_QUBITS]
    reg_ansatz = qr_ansatz[0:self.N_ANSATZ_QUBITS]
    reg_rus = qr_ansatz[self.N_ANSATZ_QUBITS:self.N_TOTAL_QUBITS]

    # Output at each time step
    cr_y = ClassicalRegister(self.n_steps, name='cr_y')
    cr_x = ClassicalRegister(self.N_RUS_QUBITS*self.n_steps, name='cr_x')
    cr_rus = ClassicalRegister(self.N_RUS_QUBITS*self.n_steps, name='cr_rus')

    # Adjustable parameters to be optimized in the learning process
    self.params = ParameterVector('P', self.N_PARAMS)
    self.rus_thetas = ParameterVector('th', self.N_THETAS)
    self.input_seq = ParameterVector('x', self.n_steps)

    # Initializing
    def initialize_circuit() -> QuantumCircuit:
        return QuantumCircuit(qr_ansatz, cr_y, cr_x, cr_rus)
    
    self.initial_state = initialize_circuit()

    def encode_angle(qc, input_data, qr_data, n_data_qubits):
        # input_data /= np.max(np.abs(input_data),axis=0)
        encoded_angle = np.arccos(input_data)
        for i in range(n_data_qubits):
            qc.ry(encoded_angle, qr_data[i])
        return qc
    
    def apply_single_qubit_gates(qc, qr_ansatz, params, n_ansatz_qubits):
        if not self.isReal:
            qc.barrier()
        for i in range(n_ansatz_qubits):
            qc.rx(params[3*i], qr_ansatz[i])
            qc.rz(params[3*i+1], qr_ansatz[i])
            qc.rx(params[3*i+2], qr_ansatz[i])
        return qc
    
    def apply_two_qubit_gates(qc, qr_ansatz, params, n_ansatz_qubits):
        if not self.isReal:
            qc.barrier()
        for i in range(-n_ansatz_qubits, 0):
            qc.cx(qr_ansatz[i], qr_ansatz[i+1])
            qc.rz(params[i], qr_ansatz[i+1])
            qc.cx(qr_ansatz[i], qr_ansatz[i+1])
        return qc
    
    def apply_ansatz(qc, qr, params, n_ansatz_qubits):
        qc = apply_single_qubit_gates(qc, qr, params[0:3*n_ansatz_qubits], n_ansatz_qubits)
        qc = apply_two_qubit_gates(qc, qr, params[3*n_ansatz_qubits:4*n_ansatz_qubits], n_ansatz_qubits)
        return qc
    
    def apply_partial_measurement(qc, qr_data, cr_y, cr_x, n_data_qubits, n_timestep):
        if not self.isReal:
            qc.barrier()
        for i in range(n_data_qubits):
            if i == 0 : qc.measure(qr_data[i], cr_y[n_timestep])
            else : qc.measure(qr_data[i], cr_x[(self.n_qubits-1)*n_timestep+i-1])
        return qc

    def apply_RUS_block_1fold(qc, qr_hid, qr_rus, cr_rus, params, n_hid_qubits, n_rus_qubits, n_timestep):
        def trial(qc, target, control, measure, theta):
            qc.ry(theta, control)
            qc.cy(control, target)
            qc.rz(-np.pi/2, control)
            qc.ry(-theta, control)
            qc.measure(control, measure)

        def repeat_block(qc, target, control, measure, theta):
            with qc.if_test((measure, 0)) as else_:
                pass
            with else_:
                qc.x(control)
                qc.ry(-np.pi/2, target)
                trial(qc, target, control, measure, theta)

        def apply_RUS(qc, target, control, measure, theta, max_trials):
            trial(qc, target, control, measure, theta)
            for _ in range(max_trials) : repeat_block(qc, target, control, measure, theta)
            return qc
        
        if not self.isReal:
            qc.barrier()
        qc.reset(qr_hid)
        qc.reset(qr_rus)
        for i in range(n_hid_qubits):
            qc = apply_RUS(qc, qr_rus[i], qr_hid[i], cr_rus[n_timestep*n_hid_qubits+i], params[i], max_trials=3)

        return qc
    
    def swap_hid_and_rus(reg_d, reg_h, reg_rus, reg_ansatz):
        _reg_h = reg_rus
        _reg_rus = reg_h
        _reg_ansatz = reg_d + _reg_h

        return (_reg_h, _reg_rus, _reg_ansatz)

    def apply_qrb(qc, reg_d, reg_h, reg_ansatz, reg_rus, cr_y, cr_x, cr_rus, params, thetas, n_data_qubits, n_ansatz_qubits, n_hid_qubits, n_rus_qubits, n_timestep):
        if n_timestep > 0:
            qc.reset(reg_d)
        qc = encode_angle(qc, self.input_seq[n_timestep], reg_d, n_data_qubits)
        qc = apply_ansatz(qc, reg_ansatz, params[4*n_timestep*n_ansatz_qubits:4*(n_timestep+1)*n_ansatz_qubits], n_ansatz_qubits)
        qc = apply_partial_measurement(qc, reg_d, cr_y, cr_x, n_data_qubits, n_timestep)
        qc = apply_RUS_block_1fold(qc, reg_h, reg_rus, cr_rus, thetas, n_hid_qubits, n_rus_qubits, n_timestep)
        return qc
    
    self.pqrnn: QuantumCircuit = initialize_circuit()
    for step in range(self.n_steps):
        self.pqrnn = apply_qrb(self.pqrnn, reg_d, reg_h, reg_ansatz, reg_rus, cr_y, cr_x, cr_rus, self.params,
                                  self.rus_thetas, self.n_qubits, self.N_ANSATZ_QUBITS, self.N_HIDDEN_QUBITS, self.N_RUS_QUBITS, step)
        (reg_h, reg_rus, reg_ansatz) = swap_hid_and_rus(reg_d, reg_h, reg_rus, reg_ansatz)


  def forward(self, input_batch: np.ndarray, params_values: np.ndarray, thetas_values: np.ndarray) -> np.ndarray:
    batch_size = len(input_batch)
    y: np.ndarray = np.zeros((batch_size,1))

    for batch_idx in range(batch_size):
        x = input_batch[batch_idx]
        pqrnn = self.pqrnn.assign_parameters({self.input_seq[i]:x[i] for i in range(self.n_steps)}, inplace=False)
        print(thetas_values)
        pqrnn.assign_parameters((params_values, thetas_values), inplace=True)

        if self.isReal:
            pqrnn = transpile(pqrnn, self.backend) #, initial_layout=initial_layout)
    
         # We run the simulation and get the counts
        counts: dict = self.backend.run(pqrnn, shots=self.n_shots).result().get_counts()
        result: float = counts['1'] / self.n_shots
        y[batch_idx][0] = result

    return y