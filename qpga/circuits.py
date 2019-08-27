import math

import squanch

from qpga import *


def cluster_state_generator(qubits):
    '''Generates a 2D cluster state on the k^2 qubits which are initially in the |00...0> state'''
    assert math.sqrt(len(qubits)).is_integer()  # must be a square number of qubits

    k = int(math.sqrt(len(qubits)))
    # TODO

    # put every qubit in |+> state
    for qubit in qubits:
        squanch.H(qubit)

    # CZ all nearest neighbors
    for row in range(k):
        for col in range(k):
            index = k * row + col
            if col + 1 < k:
                squanch.CPHASE(qubits[index], qubits[index + 1], np.pi)
                # print(f"CZ({index}, {index + 1})")
            if row + 1 < k:
                squanch.CPHASE(qubits[index], qubits[index + k], np.pi)
                # print(f"CZ({index}, {index + k})")


def QFT(qubits, num_ancillae = 0):
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
    layers = 1
    prev_gate_cz = False
    for target_index in range(num_qubits):

        # Hadamard on target requires 1 single qubit layer
        if prev_gate_cz:
            layers += 1
        prev_gate_cz = False

        # Add controlled phase gates to each subsequent qubit
        for control_index in range(target_index + 1, num_qubits):

            # If include swap, needs 3 layers per swap from qubit 0 to qubit m
            if nearest_neighbor_only:

                # Number of swaps needed to bring the control qubit adjacent to target qubit
                num_swaps = control_index - target_index - 1
                assert num_swaps >= 0

                # Move control to target
                if num_swaps > 0:
                    layers += 3 * num_swaps
                    prev_gate_cz = True  # swap ends on CZ

                # Controlled-phase of 2pi / 2^m requires 2 layers
                layers += 2
                prev_gate_cz = False

                # Move control qubit back to original position
                if num_swaps > 0:
                    layers += 3 * num_swaps
                    prev_gate_cz = True  # swap ends on CZ

            else:
                # Ignore swaps, just do controlled phase
                layers += 2

    # Include swap gates to reshuffle the output qubits to be in the same order as input
    if include_reshuffling:
        for n in range(num_qubits // 2):
            # We swap q1 with qN, q2 with qN-1, etc, which decomposes as a sequence of nearest neighbor swaps
            if nearest_neighbor_only:
                # Don't track concatenation here because each swap ends on CZ
                num_auxilliary_swaps = (num_qubits - 1 - n) - n - 1
                assert num_auxilliary_swaps >= 0, (num_auxilliary_swaps, n, num_qubits)
                layers += 3 * num_auxilliary_swaps  # swap to bring qN to q1
                layers += 3  # swap the now-neighboring target qubits
                layers += 3 * num_auxilliary_swaps  # swap to restore original ordering sans swap
            else:
                layers += 3

    # Return total count of layers
    return layers
