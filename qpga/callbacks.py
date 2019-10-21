from datetime import datetime

import h5py
from tensorflow.python.keras.callbacks import Callback

from qpga.linalg import extract_operator_from_model


class FrameWriterCallback(Callback):

    def __init__(self, input_state = None, target_state = None):
        super().__init__()
        self.input_state = input_state
        self.target_state = target_state
        self.predictions = []

    def on_batch_begin(self, batch, logs = None):
        self.predictions.append(self.model.predict(self.input_state))


class OperatorHistoryCallback(Callback):

    def __init__(self,
                 num_qubits = 0,
                 filename = None,
                 store_all_batches = False,
                 path = './',
                 in_data = None,
                 out_data = None):
        super().__init__()
        self.start_time = datetime.now()
        self.fidelities_train = []
        self.fidelities_val = []
        self.fidelity_initial = None
        self.operators = []
        self.operator_initial = None

        self.store_all_batches = store_all_batches
        self.fidelities_train_batches = []
        self.fidelities_val_batches = []
        self.operators_batches = []

        self.in_data = in_data
        self.out_data = out_data

        if filename:
            self.filename = filename
        else:
            start = self.start_time.strftime("%Y.%m.%d.%H.%M.%S")
            self.filename = path + "operator_history_{}_qubits_{}.h5".format(num_qubits, start)

    def on_train_batch_end(self, batch, logs = None):
        if self.store_all_batches:
            self.fidelities_train_batches.append(1 - logs.get('antifidelity'))
            self.operators_batches.append(extract_operator_from_model(self.model))

    def on_test_batch_end(self, batch, logs = None):
        if self.store_all_batches:
            self.fidelities_val_batches.append(1 - logs.get('val_antifidelity'))

    def on_epoch_end(self, epoch, logs = None):
        self.fidelities_train.append(1 - logs.get('antifidelity'))
        self.fidelities_val.append(1 - logs.get('val_antifidelity'))
        self.operators.append(extract_operator_from_model(self.model))

    def on_train_begin(self, logs = None):
        # basis_vecs = np.eye(2 ** self.model.num_qubits, dtype = np.complex128)
        # if not self.model.complex_inputs:
        #     basis_vecs = np_to_k_complex(basis_vecs)
        # self.fidelity_initial = self.model.evaluate(basis_vecs, callbacks=[])
        # print("Initial fidelity: {}".format(self.fidelity_initial))
        if self.in_data is not None and self.out_data is not None:
            self.fidelity_initial = self.model.evaluate(self.in_data, self.out_data)
            print("Initial fidelity: {}".format(self.fidelity_initial))
            self.operator_initial = extract_operator_from_model(self.model)

    def on_train_end(self, logs = None):
        # Save all the data to a file
        print("Writing data to {}...".format(self.filename))
        f = h5py.File(self.filename, 'w')
        f.create_dataset('fidelities_val', data = self.fidelities_val)
        f.create_dataset('fidelities_train', data = self.fidelities_train)
        f.create_dataset('fidelity_initial', data = self.fidelity_initial)
        f.create_dataset('operators', data = self.operators)
        f.create_dataset('operator_initial', data = self.operator_initial)
        if self.store_all_batches:
            f.create_dataset('fidelities_val_batches', data = self.fidelities_val_batches)
            f.create_dataset('fidelities_train_batches', data = self.fidelities_train_batches)
            f.create_dataset('operators_batches', data = self.operators_batches)
        f.close()


class StatePreparationHistoryCallback(Callback):

    def __init__(self,
                 num_qubits = 0,
                 input_state = None,
                 target_state = None,
                 filename = None,
                 groupname = None,
                 path = './'):
        super().__init__()
        self.start_time = datetime.now()
        self.input_state = input_state
        self.target_state = target_state
        self.fidelities = []
        self.output_states = []

        if filename:
            self.filename = filename
        else:
            start = self.start_time.strftime("%Y.%m.%d.%H.%M.%S")
            self.filename = path + "state_history_{}_qubits_{}.h5".format(num_qubits, start)

        self.groupname = groupname
        self.mode = 'a' if self.groupname is not None else 'w'  # if using groups, don't truncate existing file

    def on_train_begin(self, logs = None):
        # Compute initial fidelity
        if self.input_state is not None and self.target_state is not None:
            antifidelity = self.model.evaluate(self.input_state, self.target_state)
            self.fidelities.append(1 - antifidelity)

            output_state = self.model.predict(self.in_data)
            self.output_states.append(output_state)

    def on_train_batch_end(self, batch, logs = None):
        self.fidelities.append(1 - logs.get('antifidelity'))
        output_state = self.model.predict(self.input_state)
        self.output_states.append(output_state)

    def on_train_end(self, logs = None):
        # Save all the data to a file
        print("Writing data to {}...".format(self.filename))
        f = h5py.File(self.filename, self.mode)
        if self.groupname is not None:
            group = f.create_group(self.groupname)
            writeto = group
        else:
            writeto = f
        writeto.create_dataset('input_state', data = self.input_state)
        writeto.create_dataset('target_state', data = self.target_state)
        writeto.create_dataset('fidelities', data = self.fidelities)
        writeto.create_dataset('output_states', data = self.output_states)
        f.close()
