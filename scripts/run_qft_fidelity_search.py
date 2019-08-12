import sys

from tensorflow.python.keras import backend as K

sys.path.append("..")
from qpga import *
from qpga.circuits import QFT, QFT_layer_count

K.set_floatx('float64')

if __name__ == "__main__":
    N = int(sys.argv[-1])

    sys.stdout = Logger(f"QFT_fidelities_{N}_qubits.log")

    print(f"Running fidelity search for {N} qubits...")

    num_states = 2000 * N

    explicit_depth = QFT_layer_count(N, include_swap = True)
    depths = list(range(explicit_depth // 5, explicit_depth))

    in_data, out_data = prepare_training_data(QFT, N, num_states)

    fidelities = fidelity_depth_search(depths,
                                       in_data = in_data,
                                       out_data = out_data,
                                       return_on_first_convergence = True)
