import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tqdm import tqdm, tqdm_notebook


def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), m.conj().T @ m, rtol = 1e-3, atol = 1e-6)


def is_notebook():
    '''Tests to see if we are running in a jupyter notebook environment'''
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


pbar = tqdm_notebook if is_notebook() else tqdm


def k_complex_from_real(x):
    return tf.stack([x, tf.zeros_like(x)], axis = 1)


def k_real(x):
    return x[:, 0, :]


def k_imag(x):
    return x[:, 1, :]


def k_conj(x):
    '''Conjugates a concatenated complex tensor'''
    return tf.stack([k_real(x), -1 * k_imag(x)], axis = 1)


def np_to_k_complex(x):
    return np.stack([np.real(x), np.imag(x)], axis = 1)


def k_to_tf_complex(x):
    return tf.complex(k_real(x), k_imag(x))


def tf_to_k_complex(x):
    return tf.stack([tf.math.real(x), tf.math.imag(x)], axis = 1)


def np_to_complex(x):
    return np.array(k_real(x) + 1j * k_imag(x), dtype = np.complex128)


def reshape_state_vector(state):
    dim = int(np.sqrt(len(state)))
    return np.reshape(state, (dim, dim))


class FrameWriterCallback(Callback):

    def __init__(self, input_state = None, target_state = None):
        super().__init__()
        self.input_state = input_state
        self.target_state = target_state
        self.predictions = []

    def on_batch_begin(self, batch, logs = None):
        self.predictions.append(self.model.predict(self.input_state))


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering = 1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        '''
        flush method is needed for python 3 compatibility
        '''
        pass
