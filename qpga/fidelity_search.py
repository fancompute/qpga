import logging
import pickle

import numpy as np
from squanch import QStream
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import Adam

from qpga import get_random_state_vector, QPGA, antifidelity


def prepare_training_data(squanch_circuit, num_qubits, num_states):
    # Generate random state vectors
    in_states = np.array([get_random_state_vector(num_qubits) for _ in range(num_states)])
    in_states[0] = np.array([1] + [0] * (2 ** num_qubits - 1), dtype = np.complex128)  # first state should be zero qubit

    # Prepare a squanch qstream to operate on
    qstream = QStream.from_array(np.copy(in_states), use_density_matrix = False)

    # Apply the circuit to the input states
    for qsys in qstream:
        squanch_circuit(list(qsys.qubits))

    out_states = np.copy(qstream.state)

    return in_states, out_states


def fidelity_depth_search(depths, in_data, out_data,
                          log_path = None,
                          log_name = None,
                          validation_split = 0.25,
                          target_antifidelity = 1e-10,
                          learning_rate = 0.01,
                          max_attempts = 3,
                          return_on_first_convergence = True):
    '''
    Performs a sequential search to find models of given depth which can implement an operator to a given fidelity
    '''

    num_qubits = int(np.log2(in_data.shape[-1]))

    # Prepare logger output
    if log_path and log_name:
        logging.basicConfig(
                level = logging.INFO,
                format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
                handlers = [
                    logging.FileHandler("{0}/{1}".format(log_path, log_name)),
                    logging.StreamHandler()
                ])

    # log = logging.getLogger()

    fidelities = {}

    for depth in depths:
        print("\n\n\nTraining circuit of depth {} ===================================================".format(depth))
        for attempt in range(max_attempts):
            print("\n\n=> Attempt {}/{}...".format(attempt+1, max_attempts))
            model = QPGA(num_qubits, depth).as_sequential()
            model.compile(optimizer = Adam(lr = learning_rate),
                          loss = antifidelity,
                          metrics = [antifidelity])

            early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                       patience = 4,
                                                       verbose = 1,
                                                       restore_best_weights = True)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                          factor = 0.1,
                                                          patience = 2,
                                                          verbose = 1,
                                                          min_lr = 1e-6)

            history = model.fit(in_data, out_data, epochs = 25,
                                validation_split = validation_split,
                                callbacks = [early_stop, reduce_lr],
                                verbose = 2)

            antifid = history.history["antifidelity"][-1]
            fidelity = 1 - antifid

            if antifid <= target_antifidelity:
                print("Model with depth {} attained antifidelity {}.".format(depth, antifid))
                fidelities[depth] = fidelity
                with open('QFT_{}_qubits_fidelities.pickle'.format(num_qubits), 'wb') as handle:
                    pickle.dump(fidelities, handle)
                if return_on_first_convergence:
                    print("Found circuit of depth {} which converged to desired fidelity. Returning!".format(depth))
                    return fidelities
                else:
                    break
            else:
                print("Model with depth {} did not converge to target antifidelity: {}.".format(depth, antifid))
                fidelities[depth] = fidelity
                with open('QFT_{}_qubits_fidelities.pickle'.format(num_qubits), 'wb') as handle:
                    pickle.dump(fidelities, handle)

    return fidelities
