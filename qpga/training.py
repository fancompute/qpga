from tensorflow.python.keras.optimizers import Adam

from qpga import *


def build_and_train_qpga(depth, in_data, out_data,
                         validation_split = 0.1,
                         target_antifidelity = 1e-3,
                         learning_rate = 0.01,
                         max_epochs = 99,
                         batch_size = 32,
                         log_dir = None,
                         callbacks = None,
                         verbose = True,
                         print_summary = True):
    num_qubits = int(np.log2(in_data.shape[-1]))

    if verbose: print(f"Instantiating model with {num_qubits} qubits and depth of {depth}...")
    model = QPGA(num_qubits, depth).as_sequential()

    if verbose: print("Building model...")
    model.build(in_data.shape)
    if verbose: print("Done building.")

    if print_summary: print(model.summary())

    if verbose: print("Compiling model...")
    model.compile(optimizer = Adam(lr = learning_rate),
                  loss = antifidelity,
                  metrics = [antifidelity])
    if verbose: print("Done compiling.")

    if callbacks is None:
        if validation_split != 0.0:
            early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                       patience = 6,
                                                       min_delta = target_antifidelity / 10,
                                                       verbose = 1,
                                                       restore_best_weights = True)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                          factor = 0.5,
                                                          cooldown = 4,
                                                          patience = 2,
                                                          verbose = 1,
                                                          min_lr = 1e-6)
            callbacks = [early_stop, reduce_lr]
            if log_dir:
                logger = keras.callbacks.TensorBoard(log_dir = log_dir,
                                                     write_graph = True,
                                                     write_images = True,
                                                     histogram_freq = 0  # github.com/tensorflow/tensorflow/issues/30094
                                                     )
                callbacks.append(logger)
        else:
            callbacks = []

    if verbose: print("Fitting model...")
    history = model.fit(in_data, out_data,
                        epochs = max_epochs,
                        validation_split = validation_split,
                        callbacks = callbacks,
                        batch_size = batch_size,
                        verbose = 2 if verbose else 0)
    if verbose: print("Done fitting.")

    return model, history
