import sys

sys.path.append("..")

from qpga.fidelity_search import prepare_training_data
from qpga.callbacks import OperatorHistoryCallback
from qpga.circuits import QFT, QPGA, antifidelity

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

K.set_floatx('float64')

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams['figure.dpi'] = 300
np.set_printoptions(precision = 3, linewidth = 300)

if __name__ == "__main__":
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')

    N = 5
    in_data, out_data = prepare_training_data(QFT, N, 5000)

    print("Building model...")
    model = QPGA(N, 100).as_sequential()
    model.build(in_data.shape)
    model.compile(optimizer = Adam(lr = 0.0001),
                  loss = antifidelity,
                  metrics = [antifidelity])

    operator_vis = OperatorHistoryCallback(num_qubits = N, path = './logs/QFT_training')
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.5,
                                  cooldown = 4,
                                  patience = 2,
                                  verbose = 1,
                                  min_lr = 1e-6)
    callbacks = [operator_vis, reduce_lr]

    history = model.fit(in_data, out_data,
                        epochs = 300,
                        validation_split = 0.1,
                        batch_size = 64,
                        callbacks = callbacks,
                        verbose = 1)
