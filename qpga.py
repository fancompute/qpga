import tensorflow as tf
import keras
import numpy as np
# import qutip

import sys
import squanch


from keras import backend as K
from keras.models import Sequential
from keras.layers import Layer
from keras.optimizers import SGD, Adam

BS_MATRIX = 1 / np.sqrt(2) * np.array([[1 + 0j, 0 + 1j], [0 + 1j, 1 + 0j]], dtype=np.complex128)
CPHASE = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0,-1, 0],
                   [0, 0, 0, 1]], dtype=np.complex128)
IDENTITY = np.eye(2, dtype=np.complex128)

class SingleQubitOperationLayer(Layer):

    def __init__(self, num_qubits, **kwargs):
        self.num_qubits = num_qubits
        self.output_dim = 2 ** num_qubits
        super(SingleQubitOperationLayer, self).__init__(**kwargs)

    def build(self, input_shape):        
        input_dim = input_shape[-1]
        assert input_dim == self.output_dim
        
        initializer = keras.initializers.RandomUniform(minval=0, maxval=2*np.pi)

        # Create a trainable weight variable for this layer.
        self.alphas = self.add_weight(name='alphas',
                                      dtype=tf.float64,
                                      shape=(self.num_qubits,),
                                      initializer=initializer)
        self.betas  = self.add_weight(name='betas',
                                      dtype=tf.float64,
                                      shape=(self.num_qubits,),
                                      initializer=initializer)
        self.thetas = self.add_weight(name='thetas',
                                      dtype=tf.float64,
                                      shape=(self.num_qubits,),
                                      initializer=initializer)
        self.phis   = self.add_weight(name='phis',
                                      dtype=tf.float64,
                                      shape=(self.num_qubits,),
                                      initializer=initializer)
        
        self.input_shifts = phase_shifts_to_tensor_product_space(self.alphas, self.betas).to_dense()
        self.theta_shifts = phase_shifts_to_tensor_product_space(self.thetas, tf.zeros_like(self.thetas)).to_dense()
        self.phi_shifts = phase_shifts_to_tensor_product_space(self.phis, tf.zeros_like(self.phis)).to_dense()
        
        self.bs_matrix = tf.convert_to_tensor(tensors([BS_MATRIX] * self.num_qubits), dtype=tf.complex128)
        
        super(SingleQubitOperationLayer, self).build(input_shape) 

    def call(self, x):
        out = k_to_tf_complex(x)
        out = K.dot(out, self.input_shifts)
        out = K.dot(out, self.bs_matrix)
        out = K.dot(out, self.theta_shifts)
        out = K.dot(out, self.bs_matrix)
        out = K.dot(out, self.phi_shifts)
        return tf_to_k_complex(out)

    def compute_output_shape(self, input_shape):
        return input_shape


class CPhaseLayer(Layer):

    def __init__(self, num_qubits, parity=0, **kwargs):
        self.num_qubits = num_qubits
        self.parity = parity
        self.output_dim = 2 ** num_qubits
        super(CPhaseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        assert input_dim == self.output_dim
        
        ops = []
        if self.parity == 0:
            num_cphase = self.num_qubits // 2
            ops = [CPHASE] * num_cphase
            if 2 * num_cphase < self.num_qubits:
                ops.append(IDENTITY)
        else:
            ops = [IDENTITY]
            num_cphase = (self.num_qubits - 1) // 2
            ops.extend([CPHASE] * num_cphase)
            if 2 * num_cphase + 1 < self.num_qubits:
                ops.append(IDENTITY)
        
        self.transfer_matrix = tf.convert_to_tensor(tensors(ops), dtype=tf.complex128)
        
        super(CPhaseLayer, self).build(input_shape) 

    def call(self, x):
        
        out = k_to_tf_complex(x)
        
        out = K.dot(out, self.transfer_matrix)
        
        return tf_to_k_complex(out)

    def compute_output_shape(self, input_shape):
        return input_shape
    
class SingleQubitOperationLayerV2(Layer):
    def __init__(self, num_qubits, **kwargs):
        self.num_qubits = num_qubits
        self.output_dim = 2 ** num_qubits
        super(SingleQubitOperationLayerV2, self).__init__(**kwargs)

    def build(self, input_shape):        
        input_dim = input_shape[-1]
        assert input_dim == self.output_dim
        
        initializer = keras.initializers.RandomUniform(minval=0, maxval=2*np.pi)

        # Create a trainable weight variable for this layer.
        self.alphas = self.add_weight(name='alphas',
                                      dtype=tf.float64,
                                      shape=(self.num_qubits,),
                                      initializer=initializer)
        self.betas  = self.add_weight(name='betas',
                                      dtype=tf.float64,
                                      shape=(self.num_qubits,),
                                      initializer=initializer)
        self.thetas = self.add_weight(name='thetas',
                                      dtype=tf.float64,
                                      shape=(self.num_qubits,),
                                      initializer=initializer)
        self.phis   = self.add_weight(name='phis',
                                      dtype=tf.float64,
                                      shape=(self.num_qubits,),
                                      initializer=initializer)
        
        self.input_shifts = phase_shifts_to_tensor_product_space_v2(self.alphas,
                                                                    self.betas,
                                                                    self.num_qubits)
        self.theta_shifts = phase_shifts_to_tensor_product_space_v2(self.thetas,
                                                                    tf.zeros_like(self.thetas),
                                                                    self.num_qubits)
        self.phi_shifts = phase_shifts_to_tensor_product_space_v2(self.phis, tf.zeros_like(self.phis),
                                                                 self.num_qubits)
        
        self.bs_matrix = tensors([BS_MATRIX] * self.num_qubits)
        
        super(SingleQubitOperationLayerV2, self).build(input_shape) 

    def call(self, x):
        out = k_to_tf_complex(x)
        out = self.input_shifts * out
        out = out @ self.bs_matrix
        out = self.theta_shifts * out
        out = out @ self.bs_matrix
        out = self.phi_shifts * out
        return tf_to_k_complex(out)

    def compute_output_shape(self, input_shape):
        return input_shape

class QPGA(Layer):

    def __init__(self, num_qubits, num_layers, **kwargs):
        self.num_qubits = num_qubits
        self.output_dim = 2 ** num_qubits
        self.num_layers = num_layers
        super(QPGA, self).__init__(**kwargs)

    def build(self, input_shape):        
        input_dim = input_shape[-1]
        assert input_dim == self.output_dim
        
        initializer = keras.initializers.RandomUniform(minval=0, maxval=2*np.pi)

        # Create a trainable weight variable for this layer.
        self.alphas = self.add_weight(name='alphas',
                                      dtype=tf.float64,
                                      shape=(self.num_layers, self.num_qubits),
                                      initializer=initializer)
        self.betas  = self.add_weight(name='betas',
                                      dtype=tf.float64,
                                      shape=(self.num_layers, self.num_qubits),
                                      initializer=initializer)
        self.thetas = self.add_weight(name='thetas',
                                      dtype=tf.float64,
                                      shape=(self.num_layers, self.num_qubits),
                                      initializer=initializer)
        self.phis   = self.add_weight(name='phis',
                                      dtype=tf.float64,
                                      shape=(self.num_layers, self.num_qubits),
                                      initializer=initializer)
        
        self.input_shifts = qpga_phase_shifts_to_tensor_product_space(self.alphas,
                                                                      self.betas,
                                                                      self.num_qubits)
        self.theta_shifts = qpga_phase_shifts_to_tensor_product_space(self.thetas,
                                                                      tf.zeros_like(self.thetas),
                                                                      self.num_qubits)
        self.phi_shifts = qpga_phase_shifts_to_tensor_product_space(self.phis,
                                                                    tf.zeros_like(self.phis),
                                                                    self.num_qubits)
        
        self.bs_matrix = tensors([BS_MATRIX] * self.num_qubits)
        self.cphase_layers = [cphaselayer(0, self.num_qubits), cphaselayer(1, self.num_qubits)]
        
        super(QPGA, self).build(input_shape) 

    def call(self, x):
        
        out = k_to_tf_complex(x)
        
        for layer in range(self.num_layers):
            out = self.input_shifts[layer] * out
            out = out @ self.bs_matrix
            out = self.theta_shifts[layer] * out
            out = out @ self.bs_matrix
            out = self.phi_shifts[layer] * out
            out = self.cphase_layers[layer % 2] * out
        
        return tf_to_k_complex(out)

    def compute_output_shape(self, input_shape):
        return input_shape

    
### Helpers

def qpga_kronecker_phase_adder(phi_0, phi_1, num_qubits, qubit_idx):
    frequency = 2 ** (num_qubits - qubit_idx - 1)
    block_size = 2 ** qubit_idx
    num_layers = phi_0.shape[0]
    phi_0_block = tf.tile(phi_0[:, qubit_idx, tf.newaxis, tf.newaxis],
                          (1, block_size, frequency))
    phi_1_block = tf.tile(phi_1[:, qubit_idx, tf.newaxis, tf.newaxis],
                          (1, block_size, frequency))
    return tf.reshape(tf.concat((phi_0_block, phi_1_block), axis=2),
                      shape=(num_layers, 2 ** num_qubits))

def qpga_phase_shifts_to_tensor_product_space(phi_0, phi_1, num_qubits):
    phi = tf.add_n([qpga_kronecker_phase_adder(phi_0, phi_1, num_qubits, qubit_idx)
                    for qubit_idx in range(num_qubits)])
    return tf.complex(tf.cos(phi), tf.sin(phi))

def cphaselayer(parity, num_qubits):
    if parity == 0:
        num_cphase = num_qubits // 2
        ops = [CPHASE] * num_cphase
        if 2 * num_cphase < num_qubits:
            ops.append(IDENTITY)
    else:
        ops = [IDENTITY]
        num_cphase = (num_qubits - 1) // 2
        ops.extend([CPHASE] * num_cphase)
        if 2 * num_cphase + 1 < num_qubits:
            ops.append(IDENTITY)
    return np.diag(tensors(ops))

# these methods have too much setup time (more than even optimization time for N = 8!), so we do not want to have to do this "layered setup" if necessary in the final model. However, this may be nice for testing.

def build_layered_model(N, depth):
    layers = [SingleQubitOperationLayer(N)]
    for i in range(depth):
        layers.append(CPhaseLayer(N, parity=i%2))
        layers.append(SingleQubitOperationLayer(N))
    return Sequential(layers)

def build_layered_model_v2(N, depth):
    layers = [SingleQubitOperationLayerV2(N)]
    for i in range(depth):
        layers.append(CPhaseLayer(N, parity=i%2))
        layers.append(SingleQubitOperationLayerV2(N))
    return Sequential(layers)

# og phase shift to tensor product space (using LinearOperator)

def phase_shifts_to_tensor_product_space(phi_0, phi_1):
    phi_0_complex = tf.complex(tf.cos(phi_0), tf.sin(phi_0))
    phi_1_complex = tf.complex(tf.cos(phi_1), tf.sin(phi_1))

    single_qubit_ops = tf.unstack(tf.map_fn(lambda U: tf.diag(U), tf.transpose([phi_0_complex, phi_1_complex])))
    
    return tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(U) for U in single_qubit_ops])

# "kronecker phase addition" is an efficient and more gradient-friendly alternative to above method

def kronecker_phase_adder(phi_0, phi_1, num_qubits, qubit_idx):
    frequency = 2 ** (num_qubits - qubit_idx - 1)
    block_size = 2 ** qubit_idx
    return tf.reshape(tf.concat((phi_0[qubit_idx] * tf.ones((block_size, frequency), dtype=tf.float64),
                                 phi_1[qubit_idx] * tf.ones((block_size, frequency), dtype=tf.float64)), axis=1),
                      shape=(2 ** num_qubits,))

def phase_shifts_to_tensor_product_space_v2(phi_0, phi_1, num_qubits):
    phi = tf.add_n([kronecker_phase_adder(phi_0, phi_1, num_qubits, qubit_idx)
                    for qubit_idx in range(num_qubits)])
    return tf.complex(tf.cos(phi), tf.sin(phi))

# more helpers

def k_complex_from_real(x):
    return tf.stack([x, tf.zeros_like(x)], axis=1)

def k_real(x):
    return x[:, 0, :]

def k_imag(x):
    return x[:, 1, :]

def np_to_k_complex(x):
    return np.stack([np.real(x), np.imag(x)], axis=1)

def k_to_tf_complex(x):
    return tf.complex(k_real(x), k_imag(x))

def tf_to_k_complex(x):
    return tf.stack([tf.real(x), tf.imag(x)], axis=1)

def np_to_complex(x):
    return np.array(k_real(x) + 1j * k_imag(x), dtype=np.complex128)

def reshape_state_vector(state):
    dim = int(np.sqrt(len(state)))
    return np.reshape(state, (dim, dim))

def get_random_state_vector(num_wires):
    '''Returns a random 2**n-sized complex-valued state vector'''
    N = 2 ** num_wires
    mags = np.random.rand(N)
    mags /= np.linalg.norm(mags)
    phases = np.exp(1j * 2 * np.pi * np.random.rand(N))
    return mags * phases

def tensor_product(state1, state2):
    '''
    Returns the Kronecker product of two states

    :param np.array state1: the first state
    :param np.array state2: the second state
    :return: the tensor product
    '''
    if len(state1) == 0:
        return state2
    elif len(state2) == 0:
        return state1
    else:
        return np.kron(state1, state2)


def tensors(operator_list):
    '''
    Returns the iterated Kronecker product of a list of states

    :param [np.array] operator_list: list of states to tensor-product
    :return: the tensor product
    '''
    result = []
    for operator in operator_list:
        result = tensor_product(result, operator)
    return result
