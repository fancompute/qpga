import numpy as np
import squanch

from qpga.utils import np_to_k_complex, k_to_np_complex


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


def extract_operator_from_model(model, num_qubits=None):
    '''
    Reduces the action of the model to a complex-valued matrix
    :param model: the trained TensorFlow QPGA model
    :return: np.ndarray of the operator it implements
    '''
    if model.num_qubits:
        N = 2 ** model.num_qubits
    elif num_qubits is not None:
        N = 2 ** num_qubits
    else:
        raise ValueError("Need to specify num_qubits in model or args")

    basis_vecs = np.eye(N, dtype = np.complex128)
    if not model.complex_inputs:
        basis_vecs = np_to_k_complex(basis_vecs)

    output = model.predict(basis_vecs)
    if not model.complex_outputs:
        output = k_to_np_complex(output)
    operator = output

    return operator


def extract_operator_from_circuit(circuit, num_qubits):
    '''
    Reduces the action of a SQUANCH circuit to a complex-valued matrix
    :param model: a function representing the SQUANCH circuit
    :return: np.ndarray of the operator it implements
    '''
    N = 2 ** num_qubits
    basis_vecs = np.eye(N, dtype = np.complex128)
    qstream = squanch.QStream.from_array(basis_vecs, use_density_matrix = False)

    # operator = np.zeros((N, N), dtype = np.complex128)
    for qsys in qstream:
        circuit(list(qsys.qubits))

    return qstream.state
