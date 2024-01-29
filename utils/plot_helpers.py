
import numpy as np
from matplotlib import pyplot as plt
import itertools


def plot_heatmap(result_list,
                 mode,
                 plot_dims=(2, 2),
                 figsize=(7, 7),
                 ylims=(0.6, 1.0),
                 titles=('context-aware \ntrain', 'context-aware \nvalidation', 'context-unaware \ntrain', 'context-unaware \nvalidation'),
                 suptitle=None,
                 suptitle_position=1.03,
                 different_ylims=False,
                 n_runs=5,
                 matrix_indices=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
                 fontsize=18):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
    Allows for plotting multiple matrices according to plot_dims, and allows different modes:
    'max', 'min', mean', 'median', each across runs. """

    plt.figure(figsize=figsize)

    for i in range(np.prod(plot_dims)):

        if different_ylims:
            y_lim = ylims[i]
        else:
            y_lim = ylims

        heatmap = np.empty((3, 3))
        heatmap[:] = np.nan
        results = result_list[i]
        if results.shape[-1] > n_runs:
            results = results[:, :, -1]

        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        if mode == 'mean':
            values = np.nanmean(results, axis=-1)
        elif mode == 'max':
            values = np.nanmax(results, axis=-1)
        elif mode == 'min':
            values = np.nanmin(results, axis=-1)
        elif mode == 'median':
            values = np.nanmedian(results, axis=-1)

        for p, pos in enumerate(matrix_indices):
            heatmap[pos] = values[p]

        im = plt.imshow(heatmap, vmin=y_lim[0], vmax=y_lim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('# values', fontsize=fontsize)
        plt.ylabel('# attributes', fontsize=fontsize)
        plt.xticks(ticks=[0, 1, 2], labels=[4, 8, 16], fontsize=fontsize-1)
        plt.yticks(ticks=[0, 1, 2], labels=[3, 4, 5], fontsize=fontsize-1)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(y_lim)
        cbar.ax.tick_params(labelsize=fontsize-2)

        for k in range(3):
            for l in range(3):
                if not np.isnan(heatmap[l, k]):
                    ax = plt.gca()
                    _ = ax.text(l, k, np.round(heatmap[k, l], 2), ha="center", va="center", color="k",
                                fontsize=fontsize)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize+1, y=suptitle_position)

    plt.tight_layout()


def plot_heatmap_concept_x_context(result_list,
                                    mode,
                                    score,
                                    plot_dims=(2, 3),
                                    heatmap_size=(3, 3),
                                    figsize=(7, 7),
                                    ylims=(0.6, 1.0),
                                    titles=('D(3,4)', 'D(3,8)', 'D(3,16)', 'D(4,4)', 'D(4,8)', 'D(5,4)'),
                                    suptitle=None,
                                    suptitle_position=1.03,
                                    different_ylims=False,
                                    n_runs=5,
                                    matrix_indices=None,
                                    fontsize=18,
                                    inner_fontsize_fctr=0):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
    Allows for plotting multiple matrices according to plot_dims, and allows different modes:
    'max', 'min', mean', 'median', each across runs. """

    if score == 'NMI':
        score_idx = 0
    elif score == 'effectiveness':
        score_idx = 1
    elif score == 'consistency':
        score_idx = 2
    elif score == 'bosdis' or score == 'posdis':
        pass
    else:
        raise AssertionError("Score should be one of the following: 'NMI','effectiveness', 'consistency'.")

    plt.figure(figsize=figsize)

    # 6 datasets
    for i in range(np.prod(plot_dims)):
        # D(3,4), D(3,8), D(3,16)
        if i < 3:
            matrix_indices = sorted(list(itertools.product(range(3), repeat=2)), key=lambda x: x[1])
        # D(4,4), D(4,8)
        elif i == 3 or i == 4:
            matrix_indices = sorted(list(itertools.product(range(4), repeat=2)), key=lambda x: x[1])
        else:
            matrix_indices = sorted(list(itertools.product(range(5), repeat=2)), key=lambda x: x[1])

        if different_ylims:
            y_lim = ylims[i]
        else:
            y_lim = ylims

        heatmap = np.empty(heatmap_size)
        heatmap[:] = np.nan
        if score == 'bosdis' or score == 'posdis':
            results = result_list[i]
        else:
            results = result_list[score_idx][i]
            if results.shape[-1] > n_runs:
                results = results[:, :, -1]
            
        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        results_ls = [res.tolist() for res in results]

        if mode == 'mean':
            values = np.nanmean(results_ls, axis=0)
        elif mode == 'max':
            values = np.nanmax(results, axis=-1)
        elif mode == 'min':
            values = np.nanmin(results, axis=-1)
        elif mode == 'median':
            values = np.nanmedian(results, axis=-1)

        for p, pos in enumerate(matrix_indices):
            try:
                heatmap[pos] = values[p]
            except:
                IndexError

        im = plt.imshow(heatmap, vmin=y_lim[0], vmax=y_lim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('# Fixed Attributes', fontsize=fontsize)
        plt.ylabel('# Shared Attributes', fontsize=fontsize)
        plt.xticks(ticks=list(range(len(heatmap))), labels=list(range(1, len(heatmap)+1)), fontsize=fontsize-1)
        plt.yticks(ticks=list(range(len(heatmap))), labels=list(range(len(heatmap))), fontsize=fontsize-1)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(y_lim)
        cbar.ax.tick_params(labelsize=fontsize-2)

        for col in range(len(heatmap)):
            for row in range(len(heatmap[0])):
                if not np.isnan(heatmap[row, col]):
                    ax = plt.gca()
                    _ = ax.text(col, row, np.round(heatmap[row, col], 2), ha="center", va="center", color="k",
                                fontsize=fontsize+inner_fontsize_fctr)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize+1, y=suptitle_position)

    plt.tight_layout()


def plot_heatmap_different_vs(result_list,
                              mode,
                              plot_dims=(2, 2),
                              figsize=(7, 9),
                              ylims=(0.6, 1.0),
                              titles=('train', 'validation', 'zero shot objects', 'zero shot abstractions'),
                              suptitle=None,
                              suptitle_position=1.03,
                              n_runs=5,
                              matrix_indices=((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)),
                              different_ylims=False,
                              fontsize=18,
                              ):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
        Allows for plotting multiple matrices according to plot_dims, and allows different modes:
        'max', 'min', mean', 'median', each across runs.
    """

    plt.figure(figsize=figsize)

    for i in range(np.prod(plot_dims)):

        if different_ylims:
            ylim = ylims[i]
        else:
            ylim = ylims

        heatmap = np.empty((4, 2))
        heatmap[:] = np.nan
        results = result_list[i]
        if results.shape[-1] > n_runs:
            results = results[:, :, -1]

        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        if mode == 'mean':
            values = np.nanmean(results, axis=-1)
        elif mode == 'max':
            values = np.nanmax(results, axis=-1)
        elif mode == 'min':
            values = np.nanmin(results, axis=-1)
        elif mode == 'median':
            values = np.nanmedian(results, axis=-1)

        for p, pos in enumerate(matrix_indices):
            try:
                heatmap[pos] = values[p]
            except:
                continue

        im = plt.imshow(heatmap, vmin=ylim[0], vmax=ylim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('balanced', fontsize=fontsize)
        plt.ylabel('vocab size factor', fontsize=fontsize)
        plt.xticks(ticks=[0, 1], labels=['True', 'False'], fontsize=fontsize-1)
        plt.yticks(ticks=[0, 1, 2, 3], labels=[1, 2, 3, 4], fontsize=fontsize-1)
        cbar = plt.colorbar(im, fraction=0.05, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(ylim)
        cbar.ax.tick_params(labelsize=fontsize-2)

        for k in range(4):
            for l in range(2):
                if not np.isnan(heatmap[k, l]):
                    ax = plt.gca()
                    _ = ax.text(l, k, np.round(heatmap[k, l], 2), ha="center", va="center", color="k",
                                fontsize=fontsize)
        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize+1, x=0.51, y=suptitle_position)
    plt.tight_layout()


def plot_training_trajectory(results_train,
                             results_val,
                             message_length_train=None,
                             message_length_val=None,
                             steps=(1, 5),
                             figsize=(10, 7),
                             ylim=None,
                             xlim=None,
                             plot_indices=(1, 2, 3, 4, 5, 7),
                             plot_shape=(3, 3),
                             n_epochs=300,
                             train_only=False,
                             loss_plot=False,
                             message_length_plot=False,
                             titles=('D(3,4)', 'D(3,8)', 'D(3,16)', 'D(4,4)', 'D(4,8)', 'D(5,4)')):
    """ Plot the training trajectories for training and validation data"""
    plt.figure(figsize=figsize)

    for i, plot_idx in enumerate(plot_indices):
        plt.subplot(plot_shape[0], plot_shape[1], plot_idx)
        if message_length_plot:
            plt.plot(range(0, n_epochs, steps[0]), np.transpose(message_length_train[i]), color='green')
        else:
            plt.plot(range(0, n_epochs, steps[0]), np.transpose(results_train[i]), color='blue')
        if not train_only:
            plt.plot(range(0, n_epochs, steps[1]), np.transpose(results_val[i]), color='red')
            plt.legend(['train', 'val'])
            leg = plt.legend(['train', 'val'], fontsize=12)
            leg.legendHandles[0].set_color('blue')
            leg.legendHandles[1].set_color('red')
        plt.title(titles[i], fontsize=13)
        plt.xlabel('epoch', fontsize=12)
        if loss_plot:
            plt.ylabel('loss', fontsize=12)
        elif message_length_plot:
            plt.ylabel('message length', fontsize=12)
        else:
            plt.ylabel('accuracy', fontsize=12)
        if ylim:
            plt.ylim(ylim)
        if xlim:
            plt.xlim(xlim)

    if loss_plot:
        plt.suptitle('loss', x=0.53, fontsize=15)
    elif message_length_plot:
        plt.suptitle('message length', x=0.53, fontsize=15)
    else:
        plt.suptitle('accuracy', x=0.53, fontsize=15)
    plt.tight_layout()
