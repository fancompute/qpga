import numpy as np


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
