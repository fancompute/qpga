import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qutip
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def _blob(x, y, w, w_min, w_max, area, cmap = None, ax = None):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])

    handle = ax if ax is not None else plt
    # color = int(256 * (w - w_min) / (w_max - w_min))
    color = (w - w_min) / (w_max - w_min)
    handle.fill(xcorners, ycorners, color = cmap(color))


def computational_basis_labels(num_qubits, include_bras = True):
    """Creates plot labels for matrix elements in the computational basis."""
    N = 2 ** num_qubits
    basis_labels = [format(i, 'b').zfill(num_qubits) for i in range(N)]

    kets = [r"$\left|{}\right>$".format(l) for l in basis_labels]
    if include_bras:
        bras = [r"$\left<{}\right|$".format(l) for l in basis_labels]
        return [kets, bras]
    else:
        return kets


def hinton(W, xlabels = None, ylabels = None, labelsize = 9, title = None, fig = None, ax = None, cmap = None):

    if cmap is None:
        cmap = plt.get_cmap('twilight')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))

    if not (xlabels or ylabels):
        ax.axis('off')

    ax.axis('equal')
    ax.set_frame_on(False)

    height, width = W.shape
    ax.set(xlim = (0, width), ylim = (0, height))

    max_abs = np.max(np.abs(W))
    scale = 0.7

    for i in range(width):
        for j in range(height):
            x = i + 1 - 0.5
            y = j + 1 - 0.5
            _blob(x, height - y, np.angle(W[i, j]), -np.pi, np.pi,
                  np.abs(W[i, j]) / max_abs * scale, cmap = cmap, ax = ax)

    # x axis
    ax.xaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if xlabels:
        ax.set_xticklabels(xlabels, rotation = 'vertical')
        ax.xaxis.tick_top()
    ax.tick_params(axis = 'x', labelsize = labelsize, pad = 0)
    ax.xaxis.set_ticks_position('none')

    # y axis
    ax.yaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    ax.yaxis.set_ticks_position('none')
    if ylabels:
        ax.set_yticklabels(list(reversed(ylabels)))
    ax.tick_params(axis = 'y', labelsize = labelsize, pad = 0)

    # color axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '4%', pad = '2%')
    cbar = mpl.colorbar.ColorbarBase(cax, cmap = cmap,
                                     norm = mpl.colors.Normalize(-np.pi, np.pi),
                                     ticks = [])
    #                                      ticks=[-np.pi, 0, np.pi])
    cax.text(0.5, 0.0, '$-\pi$', transform = cax.transAxes, va = 'top', ha = 'center')
    cax.text(0.5, 1.0, '$+\pi$', transform = cax.transAxes, va = 'bottom', ha = 'center')
    #     cbar.ax.set_yticklabels(['$-\pi$','$0$','$+\pi$'])

    # Make title in corner
    if title is not None:
        plt.text(-.07, 1.05, title, ha = 'center', va = 'center', fontsize = 22, transform = ax.transAxes)

    return fig, ax

def loss_plot(loss_val, loss_train = None, x_units = "epochs", x_max = None, fig = None, ax = None, ylabel = None,
               ylabel_pos = 'left', log_fidelity = False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (4, 4))

    if log_fidelity:
        loss_val = np.log10(loss_val)
        if loss_train is not None:
            loss_train = np.log10(loss_train)

    ax.plot(loss_val, linestyle = '-', label = "Validation")
    ax.fill_between(np.arange(len(loss_val)), loss_val, alpha = 0.1)

    if loss_train is not None:
        ax.plot(loss_train, linestyle = ':', label = "Training")
        ax.legend(loc = 'upper left')

    if x_max is not None:
        ax.set_xlim(0, x_max - 1)
    else:
        ax.set_xlim(0, len(loss_val) - 1)

    if not log_fidelity:
        ax.set_ylim(0, 1)

    ax.yaxis.set_label_position(ylabel_pos)

    if ylabel is None:
        ylabel = "$\mathcal{F} = | \left< \psi \\right| \\tilde{U}^{\\dagger} \hat{U} \left| \psi \\right> |^2$"
    if ylabel_pos == 'left':
        ax.set_ylabel(ylabel, rotation = 90)
    else:
        ax.set_ylabel(ylabel, rotation = 270, va = 'bottom')

    if x_units == 'epochs':
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer = True))
        ax.set_xlabel("Epoch")
    elif x_units == 'iterations':
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer = True))
        ax.set_xlabel("Iteration")
    elif x_units == 'none':
        ax.set_xticks([])

    return fig, ax
