import logging
import pickle

import numpy as np
from squanch import QStream
from tensorflow.python import keras
from tensorflow.python.keras.optimizers import Adam
from tqdm import tqdm

from qpga import get_random_state_vector, QPGA, antifidelity, np_to_k_complex


def prepare_training_data(squanch_circuit, num_qubits, num_states, convert_to_k_complex=True):
    # Generate random state vectors
    in_states = np.array([get_random_state_vector(num_qubits) for _ in range(num_states)])
    in_states[0] = np.array([1] + [0] * (2 ** num_qubits - 1),
                            dtype = np.complex128)  # first state should be zero qubit

    # Prepare a squanch qstream to operate on
    qstream = QStream.from_array(np.copy(in_states), use_density_matrix = False)

    # Apply the circuit to the input states
    for qsys in tqdm(qstream):
        squanch_circuit(list(qsys.qubits))

    out_states = np.copy(qstream.state)

    if convert_to_k_complex:
        in_states = np_to_k_complex(in_states)
        out_states = np_to_k_complex(out_states)

    return in_states, out_states


def fidelity_depth_search(depths, in_data, out_data,
                          log_path = None,
                          log_name = None,
                          validation_split = 0.1,
                          target_antifidelity = 1e-10,
                          learning_rate = 0.1,
                          max_epochs = 99,
                          max_attempts = 2,
                          batch_size = 32,
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
            print(f"\n\n=> Attempt {attempt + 1}/{max_attempts}...")
            print("Instantiating model...")
            model = QPGA(num_qubits, depth).as_sequential()

            print("Building model...")
            model.build(in_data.shape)
            print("Done building.")

            print(model.summary())

            print("Compiling model...")
            model.compile(optimizer = Adam(lr = learning_rate),
                          loss = antifidelity,
                          metrics = [antifidelity])
            print("Done compiling.")

            if validation_split != 0.0:
                early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                           patience = 4,
                                                           verbose = 1,
                                                           restore_best_weights = True)
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                              factor = 0.1,
                                                              patience = 3,
                                                              verbose = 1,
                                                              min_lr = 1e-6)
                callbacks = [early_stop, reduce_lr]
            else:
                callbacks = []

            print("Fitting model...")
            history = model.fit(in_data, out_data, epochs = max_epochs,
                                validation_split = validation_split,
                                callbacks = callbacks,
                                batch_size = batch_size,
                                verbose = 2)
            print("Done fitting.")

            antifid = history.history["antifidelity"][-1]
            fidelity = 1 - antifid

            if antifid <= target_antifidelity:
                print(f"Model with depth {depth} attained antifidelity {antifid}.")
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
                fidelities[depth] = fidelity
                with open(f'QFT_{num_qubits}_qubits_fidelities.pickle', 'wb') as handle:
                    pickle.dump(fidelities, handle)

    return fidelities
