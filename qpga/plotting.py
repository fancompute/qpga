import matplotlib.pyplot as plt
import numpy as np
import qutip

from qpga.utils import reshape_state_vector


def plot_state_comparison(true_state, pred_state, iteration = None, savefig = False):
    # true_state = np_to_complex(true_state)[0]
    # pred_state = np_to_complex(pred_state)[0]
    fidelity = np.abs(np.dot(true_state.conj(), pred_state)) ** 2
    mat_true = reshape_state_vector(true_state)
    mat_pred = reshape_state_vector(pred_state)
    fig = plt.figure(figsize = (18, 6))
    fig.text(.5, .85, "Fidelity: {:.4f}".format(fidelity), fontsize = 14, ha = 'center', va = 'center')
    if iteration is not None:
        fig.text(.83, .13, "Iteration: {}".format(iteration))
    ax1 = fig.add_subplot(121, projection = '3d')
    ax2 = fig.add_subplot(122, projection = '3d')
    qutip.matrix_histogram_complex(mat_true, xlabels = [''], ylabels = [''], title = "Target state", fig = fig,
                                   ax = ax1)
    qutip.matrix_histogram_complex(mat_pred, xlabels = [''], ylabels = [''], title = "Predicted state", fig = fig,
                                   ax = ax2)
    if savefig:
        title = str(iteration).zfill(5)
        plt.savefig(f"frames/{title}.png", dpi = 144)
        plt.close()
    else:
        plt.show()
