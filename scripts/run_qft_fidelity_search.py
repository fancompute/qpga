import argparse
import sys
from datetime import datetime

sys.path.append("..")
from qpga import *
from qpga.circuits import QFT, QFT_layer_count
from qpga.utils import *
from qpga.fidelity_search import *

# dynamically grow GPU memory to prevent tf from just claiming all of it at once
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config = config))

DEFAULT_BATCH_SIZE = 128 if tf.test.is_gpu_available() else 32

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run a fidelity search on QPGA with specified number of qubits")
    parser.add_argument('--num_qubits', type = int, default = 4)
    parser.add_argument('--start', type = int)
    parser.add_argument('--num_states', type = int)
    parser.add_argument('--num_ancillae', type = int, default = 0)
    parser.add_argument('--batch_size', type = int, default = DEFAULT_BATCH_SIZE)
    parser.add_argument('--max_attempts', type = int)
    parser.add_argument('--target_antifidelity', type = float, default = 1e-3)

    args = parser.parse_args()
    N = args.num_qubits

    start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    log_path = "./logs"
    experiment = "QFT"
    size = f"{N}_qubits"
    if args.num_ancillae > 0:
        size += f"_{args.num_ancillae}_ancillae"
    run = f"run_{start_time}"
    filename = "console.log"

    path = f"{log_path}/{experiment}/{size}/{run}"

    sys.stdout = Logger(f"{path}/{filename}")

    print(f"Running fidelity search for {N} qubits...")
    print(f"args: {args}")

    if args.num_states:
        num_states = args.num_states
    else:
        num_states = 2000 * N

    explicit_depth = QFT_layer_count(N, nearest_neighbor_only = True)

    if args.start:
        depths = list(range(args.start, explicit_depth))
    else:
        depths = list(range(explicit_depth // 4, explicit_depth))

    if args.max_attempts:
        max_attempts = args.max_attempts
    else:
        size = N + args.num_ancillae
        if size <= 4:
            max_attempts = 4
        elif size <= 6:
            max_attempts = 3
        else:
            max_attempts = 2

    print(f"Preparing {num_states} training states...")

    in_data, out_data = prepare_training_data(lambda qubits: QFT(qubits, num_ancillae = args.num_ancillae),
                                              N + args.num_ancillae, num_states)
    print(f"Done! \n\n")

    fidelities = fidelity_depth_search(depths, in_data, out_data, path,
                                       batch_size = args.batch_size,
                                       target_antifidelity = args.target_antifidelity,
                                       max_attempts = max_attempts,
                                       return_on_first_convergence = True)
