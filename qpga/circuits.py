import squanch

from qpga import *


def QFT(qubits):
    '''Applies quantum Fourier transform to inputs'''
    N = len(qubits)
    for n in range(N, 0, -1):
        target = qubits[N - n]
        squanch.H(target)
        for m in range(1, n):
            squanch.CPHASE(qubits[N - n + m], target, 2 * np.pi / 2 ** (m + 1))


def QFT_layer_count(num_qubits, include_swap = True):
    '''
    Counts the number of layers (1 layer = single qubit + CZ) required to
    implement the n-qubit quantum Fourier transform. If include_swap set to
    true, include gates to allow qubit A to talk to non-neighbor qubit B
    '''
    layers = 0
    can_concat_single_qubit_op = False
    for n in range(num_qubits, 0, -1):

        # Hadamard on target requires 1 single qubit layer
        if can_concat_single_qubit_op:
            layers -= 1
        layers += 1
        can_concat_single_qubit_op = True

        # If include swap, needs 3 layers per swap from qubit 0 to qubit m
        if include_swap:
            distance = n - 1
            if distance > 0:
                if can_concat_single_qubit_op:
                    layers -= 1  # first layer of swap can concat with previous H
                layers += 3 * distance
                can_concat_single_qubit_op = False  # swap ends on CZ

        # Controlled-phase of 2pi / 2^m requires 2 layers
        if can_concat_single_qubit_op:
            layers -= 1
        layers += 2
        can_concat_single_qubit_op = True

        # Include swap gates back to original position
        if include_swap:
            distance = n - 1
            if distance > 0:
                if can_concat_single_qubit_op:
                    layers -= 1  # first layer of swap can concat with previous H
                layers += 3 * distance
                can_concat_single_qubit_op = False  # swap ends on CZ

    # Return total count of layers
    return layers
