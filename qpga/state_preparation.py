import numpy as np


def get_basis_state(n, num_qubits):
    '''Gets the nth computational basis vector in the 2^num_qubits dimensional space'''
    assert n < 2 ** num_qubits
    return np.array(list(map(int, format(n, 'b').zfill(num_qubits))), dtype=np.complex128)


def get_random_state_vector(num_qubits):
    '''Returns a random 2**n-sized complex-valued state vector'''
    N = 2 ** num_qubits
    mags = np.random.rand(N)
    mags /= np.linalg.norm(mags)
    phases = np.exp(1j * 2 * np.pi * np.random.rand(N))
    return mags * phases


def noon_state(num_qubits):
    '''Prepares an n-qubit state in 1/sqrt2 * (|00...0> + |11...1>)'''
    return 1 / np.sqrt(2) * np.array([1] + [0] * (2 ** num_qubits - 2) + [1], dtype = np.complex128)


def zero_state(num_qubits):
    '''Prepares the n-qubit initial state |00...0>'''
    return np.array([1] + [0] * (2 ** num_qubits - 1), dtype = np.complex128)
