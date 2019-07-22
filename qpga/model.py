import keras
import numpy as np
import tensorflow as tf
from keras import backend as K, Sequential
from keras.layers import Layer, Lambda

from qpga.constants import IDENTITY, CPHASE_MOD, BS_MATRIX, CPHASE
from qpga.linalg import tensors
from qpga.utils import k_to_tf_complex, tf_to_k_complex


def phase_shifts_to_tensor_product_space(phi_0, phi_1):
    phi_0_complex = tf.complex(tf.cos(phi_0), tf.sin(phi_0))
    phi_1_complex = tf.complex(tf.cos(phi_1), tf.sin(phi_1))

    single_qubit_ops = tf.unstack(tf.map_fn(lambda U: tf.linalg.tensor_diag(U), tf.transpose([phi_0_complex, phi_1_complex])))

    return tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(U) for U in single_qubit_ops])


class SingleQubitOperationLayer(Layer):

    def __init__(self, num_qubits, **kwargs):
        self.num_qubits = num_qubits
        self.output_dim = 2 ** num_qubits
        super(SingleQubitOperationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        assert input_dim == self.output_dim

        initializer = keras.initializers.RandomUniform(minval = 0, maxval = 2 * np.pi)

        # Create a trainable weight variable for this layer.
        self.alphas = self.add_weight(name = 'alphas',
                                      dtype = tf.float64,
                                      shape = (self.num_qubits,),
                                      initializer = initializer)
        self.betas = self.add_weight(name = 'betas',
                                     dtype = tf.float64,
                                     shape = (self.num_qubits,),
                                     initializer = initializer)
        self.thetas = self.add_weight(name = 'thetas',
                                      dtype = tf.float64,
                                      shape = (self.num_qubits,),
                                      initializer = initializer)
        self.phis = self.add_weight(name = 'phis',
                                    dtype = tf.float64,
                                    shape = (self.num_qubits,),
                                    initializer = initializer)

        self.input_shifts = phase_shifts_to_tensor_product_space(self.alphas, self.betas).to_dense()
        self.theta_shifts = phase_shifts_to_tensor_product_space(self.thetas, tf.zeros_like(self.thetas)).to_dense()
        self.phi_shifts = phase_shifts_to_tensor_product_space(self.phis, tf.zeros_like(self.phis)).to_dense()

        self.bs_matrix = tf.convert_to_tensor(tensors([BS_MATRIX] * self.num_qubits), dtype = tf.complex128)

        super(SingleQubitOperationLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        out = x

        # out = k_to_tf_complex(out)

        out = K.dot(out, self.input_shifts)
        out = K.dot(out, self.bs_matrix)
        out = K.dot(out, self.theta_shifts)
        out = K.dot(out, self.bs_matrix)
        out = K.dot(out, self.phi_shifts)

        # out = tf_to_k_complex(out)

        return out

    def compute_output_shape(self, input_shape):
        return input_shape


class CPhaseLayer(Layer):

    def __init__(self, num_qubits, parity = 0, use_standard_cphase = False, **kwargs):
        self.num_qubits = num_qubits
        self.parity = parity
        self.use_standard_cphase = use_standard_cphase
        self.output_dim = 2 ** num_qubits
        super(CPhaseLayer, self).__init__(**kwargs)

    def get_cphase_gate(self):
        return CPHASE if self.use_standard_cphase else CPHASE_MOD

    def build(self, input_shape):
        input_dim = input_shape[-1]
        assert input_dim == self.output_dim

        ops = []
        if self.parity == 0:
            num_cphase = self.num_qubits // 2
            for _ in range(num_cphase):
                ops.append(self.get_cphase_gate())
            if 2 * num_cphase < self.num_qubits:
                ops.append(IDENTITY)
        else:
            ops = [IDENTITY]
            num_cphase = (self.num_qubits - 1) // 2
            for _ in range(num_cphase):
                ops.append(self.get_cphase_gate())
            if 2 * num_cphase + 1 < self.num_qubits:
                ops.append(IDENTITY)

        self.transfer_matrix_np = tensors(ops)
        self.transfer_matrix = tf.convert_to_tensor(self.transfer_matrix_np, dtype = tf.complex128)

        super(CPhaseLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        out = x

        # out = k_to_tf_complex(x)

        out = K.dot(out, self.transfer_matrix)

        # out = tf_to_k_complex(out)

        return out

    def compute_output_shape(self, input_shape):
        return input_shape


class QPGA(keras.Model):

    def __init__(self, num_qubits, depth,
                 complex_inputs = False,
                 complex_outputs = False,
                 use_standard_cphase = True):
        super(QPGA, self).__init__(name = 'qpga')

        self.num_qubits = num_qubits
        self.depth = depth
        self.complex_inputs = complex_inputs
        self.complex_outputs = complex_outputs
        self.use_standard_cphase = use_standard_cphase

        self.input_layer = SingleQubitOperationLayer(self.num_qubits)
        self.single_qubit_layers = []
        self.cphase_layers = []
        for i in range(depth):
            self.cphase_layers.append(CPhaseLayer(self.num_qubits,
                                                  parity = i % 2,
                                                  use_standard_cphase = self.use_standard_cphase))
            self.single_qubit_layers.append(SingleQubitOperationLayer(self.num_qubits))

    def as_sequential(self):
        '''Converts the QPGA instance into a sequential model for easier inspection'''
        model = Sequential()

        if not self.complex_inputs:
            model.add(Lambda(lambda x: k_to_tf_complex(x)))

        model.add(self.input_layer)
        for cphase_layer, single_qubit_layer in zip(self.cphase_layers, self.single_qubit_layers):
            model.add(cphase_layer)
            model.add(single_qubit_layer)

        if not self.complex_outputs:
            model.add(Lambda(lambda x: tf_to_k_complex(x)))

        return model

    def call(self, inputs):
        x = inputs

        if not self.complex_inputs:
            x = k_to_tf_complex(x)

        x = self.input_layer(x)
        for cphase_layer, single_qubit_layer in zip(self.cphase_layers, self.single_qubit_layers):
            x = cphase_layer(x)
            x = single_qubit_layer(x)

        if not self.complex_outputs:
            x = tf_to_k_complex(x)

        return x


def antifidelity(state_true, state_pred):
    # inner_prods = tf.einsum('bs,bs->b', tf.math.conj(state_true), state_pred)
    state_true = k_to_tf_complex(state_true)
    state_pred = k_to_tf_complex(state_pred)
    inner_prods = tf.reduce_sum(tf.multiply(tf.math.conj(state_true), state_pred), 1)
    amplitudes = tf.abs(inner_prods)
    return tf.ones_like(amplitudes) - amplitudes ** 2
