import squanch

from qpga import *
import math

def cluster_state_generator(qubits):
    '''Generates a 2D cluster state on the k^2 qubits'''
    assert math.sqrt(len(qubits)).is_integer() # must be a square number of qubits

    k = int(math.sqrt(len(qubits)))
    # TODO





    pass








def QFT(qubits, num_ancillae=0):
    '''Applies quantum Fourier transform to inputs, then shuffles the output to be in the same order as input'''
    N = len(qubits) - num_ancillae
    for n in range(N):
        target = qubits[n]
        squanch.H(target)
        for m in range(1, N - n):
            squanch.CPHASE(qubits[n + m], target, 2 * np.pi / (2 ** (m + 1)))
    # At this point, the output bits are in reverse order, so swap first with last and so on to get right order
    for n in range(N // 2):
        squanch.SWAP(qubits[n], qubits[N - n - 1])


def QFT_layer_count(num_qubits, nearest_neighbor_only = True, include_reshuffling = True):
    '''
    Counts the number of layers (1 layer = single qubit + CZ) required to
    implement the n-qubit quantum Fourier transform. If nearest_neighbor_only set to
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
        if nearest_neighbor_only:
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
        if nearest_neighbor_only:
            distance = n - 1
            if distance > 0:
                if can_concat_single_qubit_op:
                    layers -= 1  # first layer of swap can concat with previous H
                layers += 3 * distance
                can_concat_single_qubit_op = False  # swap ends on CZ

    # Include swap gates to reshuffle the output qubits to be in the same order as input
    if include_reshuffling:
        for n in range(num_qubits//2):
            # We swap q1 with qN, q2 with qN-1, etc, which decomposes as a sequence of nearest neighbor swaps
            if nearest_neighbor_only:
                # Don't track concatenation here because each swap ends on CZ
                num_auxilliary_swaps = (num_qubits - 1 - n) - n - 1
                assert num_auxilliary_swaps >= 0, (num_auxilliary_swaps, n, num_qubits)
                layers += 3 * num_auxilliary_swaps # swap to bring qN to q1
                layers += 3 # swap the now-neighboring target qubits
                layers += 3 * num_auxilliary_swaps # swap to restore original ordering sans swap
            else:
                layers += 3

    # Return total count of layers
    return layers
