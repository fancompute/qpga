import pickle
from pprint import pprint

import numpy as np
from squanch import QStream
from tensorflow.python import keras
from tqdm import tqdm

from qpga.state_preparation import get_random_state_vector
from qpga.utils import np_to_k_complex
from qpga.training import build_and_train_qpga


def prepare_training_data(squanch_circuit, num_qubits, num_states, convert_to_k_complex = True):
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


def fidelity_depth_search(depths, in_data, out_data, path,
                          validation_split = 0.1,
                          target_antifidelity = 1e-3,
                          learning_rate = 0.01,
                          max_epochs = 99,
                          max_attempts = 2,
                          batch_size = 32,
                          return_on_first_convergence = True,
                          save_successful_model = True):
    '''
    Performs a sequential search to find models of given depth which can implement an operator to a given fidelity
    '''

    num_qubits = int(np.log2(in_data.shape[-1]))

    # # Prepare logger output
    # if log_path and experiment_name:
    #     logging.basicConfig(
    #             level = logging.INFO,
    #             format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    #             handlers = [
    #                 logging.FileHandler("{0}/{1}".format(log_path, experiment_name)),
    #                 logging.StreamHandler()
    #             ])

    # log = logging.getLogger()

    fidelities = {}

    for depth in depths:
        print(f"\n\n\nTraining circuit of depth {depth} =============================================================")
        for attempt in range(max_attempts):
            print(f"\n\n=> Attempt {attempt + 1}/{max_attempts}...")

            log_dir = f"{path}/tensorboard/depth_{depth}_attempt{attempt}"

            model, history = build_and_train_qpga(depth, in_data, out_data,
                                                  validation_split = validation_split,
                                                  target_antifidelity = target_antifidelity,
                                                  learning_rate = learning_rate,
                                                  max_epochs = max_epochs,
                                                  batch_size = batch_size,
                                                  log_dir = log_dir)

            antifid = history.history["antifidelity"][-1]
            fidelity = 1 - antifid

            if antifid <= target_antifidelity:
                print(f"Model with depth {depth} attained antifidelity {antifid}.")

                if save_successful_model:
                    filename = f"{path}/model.h5"
                    print(f"Saving model to {filename}...")
                    keras.models.save_model(model, filename,
                                            overwrite = True,
                                            include_optimizer = True,
                                            save_format = 'h5')
                    print("Model config: ")
                    pprint(model.get_config())

                fidelities[depth] = fidelity
                with open(f'{path}/QFT_{num_qubits}_qubits_fidelities.pickle', 'wb') as handle:
                    pickle.dump(fidelities, handle)
                if return_on_first_convergence:
                    print(f"Found circuit of depth {depth} which converged to desired fidelity. Aborting search.")
                    return fidelities
                else:
                    break
            else:
                print(f"Model with depth {depth} did not converge to target antifidelity: {antifid}.")
                fidelities[depth] = fidelity
                with open(f'{path}/QFT_{num_qubits}_qubits_fidelities.pickle', 'wb') as handle:
                    pickle.dump(fidelities, handle)

    return fidelities
