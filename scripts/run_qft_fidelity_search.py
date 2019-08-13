import sys
import argparse

from tensorflow.python.keras import backend as K

sys.path.append("..")
from qpga import *
from qpga.circuits import QFT, QFT_layer_count

K.set_floatx('float64')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a fidelity search on QPGA with specified number of qubits")
    parser.add_argument('num_qubits', type=int)
    parser.add_argument('--start', type=int)
    parser.add_argument('--num_states', type=int)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--max_attempts', type=int, default = 2)


    args = parser.parse_args()
    N = args.num_qubits

    sys.stdout = Logger(f"QFT_fidelities_{N}_qubits.log")

    print(f"Running fidelity search for {N} qubits...")

    if args.num_states :
        num_states = args.num_states
    else:
        num_states = 2000 * N

    explicit_depth = QFT_layer_count(N, include_swap = True)

    if args.start:
        depths = list(range(args.start, explicit_depth))
    else:
        depths = list(range(explicit_depth // 5, explicit_depth))

    print(f"Preparing {num_states} training states...")
    in_data, out_data = prepare_training_data(QFT, N, num_states)
    print(f"Done! \n\n")

    fidelities = fidelity_depth_search(depths,
                                       in_data = in_data,
                                       out_data = out_data,
                                       batch_size = args.batch_size,
                                       max_attempts = args.max_attempts,
                                       return_on_first_convergence = True)
