import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

from qpga.constants import IDENTITY, CPHASE_MOD
from qpga.linalg import tensors
from qpga.utils import k_to_tf_complex, tf_to_k_complex


def phase_shifts_to_tensor_product_space(phi_0, phi_1):
    phi_0_complex = tf.complex(tf.cos(phi_0), tf.sin(phi_0))
    phi_1_complex = tf.complex(tf.cos(phi_1), tf.sin(phi_1))

    single_qubit_ops = tf.unstack(tf.map_fn(lambda U: tf.diag(U), tf.transpose([phi_0_complex, phi_1_complex])))

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

    def __init__(self, num_qubits, parity = 0, **kwargs):
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
            ops = [CPHASE_MOD] * num_cphase
            if 2 * num_cphase < self.num_qubits:
                ops.append(IDENTITY)
        else:
            ops = [IDENTITY]
            num_cphase = (self.num_qubits - 1) // 2
            ops.extend([CPHASE_MOD] * num_cphase)
            if 2 * num_cphase + 1 < self.num_qubits:
                ops.append(IDENTITY)

        self.transfer_matrix = tf.convert_to_tensor(tensors(ops), dtype = tf.complex128)

        super(CPhaseLayer, self).build(input_shape)

    def call(self, x):

        out = k_to_tf_complex(x)

        out = K.dot(out, self.transfer_matrix)

        return tf_to_k_complex(out)

    def compute_output_shape(self, input_shape):
        return input_shape
