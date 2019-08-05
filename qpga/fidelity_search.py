import logging
import pickle

import numpy as np
from squanch import QStream
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import Adam

from qpga import get_random_state_vector, np_to_k_complex, QPGA, antifidelity


def prepare_training_data(squanch_circuit, num_qubits, num_states):
    # Generate random state vectors
    states = np.array([get_random_state_vector(num_qubits) for _ in range(num_states)])
    states[0] = np.array([1] + [0] * (2 ** num_qubits - 1), dtype = np.complex128)  # first state should be zero qubit

    # Prepare a squanch qstream to operate on
    qstream = QStream.from_array(np.copy(states), use_density_matrix = False)

    # Apply the circuit to the input states
    for qsys in qstream:
        squanch_circuit(list(qsys.qubits))

    out_states = np.copy(qstream.state)

    # Return states as tf tensors
    in_data = np_to_k_complex(states)
    out_data = np_to_k_complex(out_states)

    return in_data, out_data


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
        print(f"\n\n\nTraining circuit of depth {depth} =============================================================")
        for attempt in range(max_attempts):
            print(f"Attempt {attempt}/{max_attempts}...")
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

            if antifid <= target_antifidelity:
                print(f"Model with depth {depth} attained antifidelity {antifid}.")
                fidelity = 1 - antifid
                fidelities[depth] = fidelity
                with open(f'QFT_{num_qubits}_qubits_fidelities.pickle', 'wb') as handle:
                    pickle.dump(fidelities, handle)
                if return_on_first_convergence:
                    print(f"Found circuit of depth {depth} which converged to desired fidelity. Returning!")
                    return fidelities
                else:
                    break
            else:
                print(f"Model with depth {depth} did not converge to target antifidelity: {antifid}.")
                fidelities[depth] = None
                with open(f'QFT_{num_qubits}_qubits_fidelities.pickle', 'wb') as handle:
                    pickle.dump(fidelities, handle)

    return fidelities
