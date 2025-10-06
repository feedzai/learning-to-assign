import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import colors as m_colors
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve

from .eda import hist_of_means
from .utils import get_colorblind_cat_cmap


def plot_confusion_matrix(y_true, y_pred, normalize=None, return_matrix=False, labels=None, cmap='Blues', colorbar=False):
    if labels is None:
        labels = [0, 1]
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels, normalize=normalize)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    disp.plot(cmap=cmap, colorbar=colorbar)
    plt.show()

    if return_matrix:
        return conf_matrix

def plot_roc_curve(y_true, y_score, color='steelblue', title=None, show=True):
    """
    Plot several ROC curves.
    :param y_true: true labels array.
    :param y_score: scores array.
    :param color: matplotlib color for the line.
    :param title: plot title. None for no title.
    """
    diag = np.arange(0, 1 + 0.01, 0.01)
    plt.plot(diag, diag, c='grey', linestyle='dashed')

    fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_score)
    ax = plt.plot(fpr, tpr, c=color)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    else:
        return ax


def plot_roc_curves(y_true, scores_dict, title="ROC Curves", legend=True, show=True):
    """
    Plot several ROC curves with the same underlying labels.
    :param y_true: true labels array.
    :param scores_dict: dictionary where keys are model names and values are the score arrays.
    :param title: plot title. None for no title.
    :param legend: show legend
    :param show: plt.show(). If false, returns axis object.
    """
    diag = np.arange(0, 1 + 0.01, 0.01)
    ax = plt.plot(diag, diag, c='grey', linestyle='dashed')

    cmap = get_colorblind_cat_cmap(cmap='wong')
    i = 0
    for model, scores_arr in scores_dict.items():
        fpr, tpr, thr = roc_curve(y_true=y_true, y_score=scores_arr)
        plt.plot(fpr, tpr, label=model, c=cmap[i])
        i += 1

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if legend:
        plt.legend(loc='lower right')
    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    else:
        return ax

def plot_independent_roc_curves(labels_dict, scores_dict, hue_dict=None, plot_average_roc=False,
                                title=None, alpha=1,
                                palette='Blues', uniform_color=False,
                                show=True):
    """
    Plot several ROC curves with diferrent underlying labels.
    :param labels_dict: dictionary where keys are model names and values are the label arrays.
    :param scores_dict: dictionary where keys are model names and values are the score arrays.
    :param hue_dict: dictionary where keys are model names and values are the hue-encoded values.
    :param plot_average_roc: boolean to indicate whether to plot the average ROC curve.
    :param title: plot title. None for no title.
    :param alpha: matplotlib alpha parameter (transparency).
    :param palette: matplotlib-compatible palette.
    :param uniform_color: when hue_dict is None, use this color.
    :param show: plt.show(). If false, returns axis object.
    """
    if hue_dict is None and uniform_color is False:  # no color mapping and no specified color
        uniform_color = 'steelblue'

    if hue_dict is not None and uniform_color is not False:
        raise ValueError("Either hue_dict is None (not color encoding) or uniform_color is False (color encoding)")

    if plot_average_roc:
        ax = plot_average_roc_curve(
            labels_dict=labels_dict, scores_dict=scores_dict,
            linewidth=2, label='Average ROC',
            show=False,
        )
    else:
        diag = np.arange(0, 1 + 0.01, 0.01)
        plt.plot(diag, diag, c='grey', linestyle='dashed')

    keys_labels = list(labels_dict.keys())
    keys_scores = list(scores_dict.keys())
    assert set(keys_labels) == set(keys_scores)
    keys = keys_labels

    try:
        is_uniform_encoding = isinstance(list(hue_dict.values())[0], str)
    except AttributeError:
        is_uniform_encoding = False

    if hue_dict is not None and not is_uniform_encoding:
        cmap = cm.get_cmap(palette)
        norm = m_colors.Normalize(vmin=0.9 * min(hue_dict.values()), vmax=1.1 * max(hue_dict.values()))

    for key in keys:
        if labels_dict[key].size == 0:  # empty
            continue

        fpr_arr, tpr_arr, thr = roc_curve(
            y_true=labels_dict[key],
            y_score=scores_dict[key]
        )

        if hue_dict is None:
            key_color = uniform_color
        elif is_uniform_encoding:  # uniform color encoding for each ROC curve
            key_color = hue_dict[key]
        else:  # variable color encoding within each ROC curve
            key_color = cmap(norm(hue_dict[key]))

        plt.plot(
            fpr_arr, tpr_arr,
            c=key_color, alpha=alpha,
            label=key if is_uniform_encoding else None,
        )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    if plot_average_roc or is_uniform_encoding:
        plt.legend()
    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    else:
        return ax

def plot_average_roc_curve(labels_dict, scores_dict, title=None, color='darkblue', show=True, **kwargs):
    """
    Plot average ROC curve.
    :param labels_dict: dictionary where keys are model names and values are the label arrays.
    :param scores_dict: dictionary where keys are model names and values are the score arrays.
    :param title: plot title. None for no title.
    :param color: string passed to plt.plot.
    :param show: plt.show(). If false, returns axis object.
    :param kwargs: passed to plt.plot.
    """
    keys_labels = list(labels_dict.keys())
    keys_scores = list(scores_dict.keys())
    assert set(keys_labels) == set(keys_scores)
    keys = keys_labels

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for key in keys:
        viz = RocCurveDisplay.from_predictions(y_true=labels_dict[key], y_pred=scores_dict[key])
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        plt.close()

    mean_tpr = np.mean(tprs, axis=0)

    diag = np.arange(0, 1 + 0.01, 0.01)
    fig, ax = plt.subplots()
    plt.plot(diag, diag, c='grey', linestyle='dashed')
    plt.plot(mean_fpr, mean_tpr, color=color, **kwargs)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    else:
        return ax

def plot_multiclass_roc_curves(y_true, y_score, ordered_classes, show=True, **kwargs):
    """
    Plot one ROC curve per class.
    :param y_true: true labels array.
    :param y_score: score 2D array.
    :param ordered_classes: target classes ordered in the same way as in the model (model.classes_).
    :param show: plt.show(). If false, returns axis object.
    :param kwargs: plot_independent_roc_curves kwargs.
    """
    class_scores_dict = dict()
    class_binary_labels_dict = dict()
    for ix, c in enumerate(ordered_classes):
        class_scores_dict[c] = y_score[:, ix]
        class_binary_labels_dict[c] = (y_true == c).astype(int)

    ax = plot_independent_roc_curves(
        labels_dict=class_binary_labels_dict,
        scores_dict=class_scores_dict,
        show=False,
        **kwargs
    )

    if show:
        plt.show()
    else:
        return ax

def plot_recall_at_top_k_curves(y_true, scores_dict, title=None, show=True):
    """
    Plot several recall @ top-k curves.
    :param y_true: true labels array.
    :param scores_dict: dictionary where keys are model names and values are the score arrays.
    :param title: plot title. None for no title.
    :param show: plt.show(). If false, returns axis object.
    """
    no_info_x_y = np.arange(0, 1 + 0.01, 0.01)
    plt.plot(no_info_x_y, no_info_x_y, c='grey', linestyle='dashed')

    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    cmap = get_colorblind_cat_cmap(cmap='wong')
    i = 0
    for model, scores_arr in scores_dict.items():
        if isinstance(scores_arr, pd.Series):
            scores_arr = scores_arr.values

        k_arr = np.arange(0, 1 + 0.01, 0.01)
        sort_ix = np.flip(np.argsort(scores_arr))  # argsort is always ascending
        sorted_y_true = y_true[sort_ix]
        lp = (y_true == 1).sum()

        recall_list = list()
        for k in k_arr:
            k_n = int(k * y_true.shape[0])
            tp = (sorted_y_true[:k_n] == 1).sum()
            recall = tp / lp
            recall_list.append(recall)

        recall_arr = np.array(recall_list)

        plt.plot(k_arr, recall_arr, label=model, c=cmap[i])
        i += 1

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('k (%)')
    plt.ylabel('Recall')
    plt.legend()
    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    else:
        return ax


def plot_precision_at_top_k_curves(y_true, scores_dict, title=None, show=True):
    """
    Plot several precision @ top-k curves.
    :param y_true: true labels array.
    :param scores_dict: dictionary where keys are model names and values are the score arrays.
    :param title: plot title. None for no title.
    :param show: plt.show(). If false, returns axis object.
    """
    prevalence = (y_true == 1).sum() / y_true.shape[0]
    no_info_x = np.arange(0 + 0.01, 1 + 0.01, 0.01)
    no_info_y = np.full(shape=no_info_x.shape, fill_value=prevalence)
    ax = plt.plot(no_info_x, no_info_y, c='grey', linestyle='dashed')

    if isinstance(y_true, pd.Series):
        y_true = y_true.values

    cmap = get_colorblind_cat_cmap(cmap='wong')
    i = 0
    for model, scores_arr in scores_dict.items():
        if isinstance(scores_arr, pd.Series):
            scores_arr = scores_arr.values

        k_arr = np.arange(0 + 0.01, 1 + 0.01, 0.01)
        sort_ix = np.flip(np.argsort(scores_arr))  # argsort is always ascending
        sorted_y_true = y_true[sort_ix]

        precision_list = list()
        for k in k_arr:
            k_n = int(k * y_true.shape[0])
            tp = (sorted_y_true[:k_n] == 1).sum()
            precision = tp / k_n
            precision_list.append(precision)

        precision_arr = np.array(precision_list)

        plt.plot(k_arr, precision_arr, label=model, c=cmap[i])
        i += 1

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('k (%)')
    plt.ylabel('Precision')
    plt.legend()
    if title is not None:
        plt.title(title)

    if show:
        plt.show()
    else:
        return ax

def plot_calibration(y_true, y_score, bins=50, show=True):
    """
    Plotting true probability of decline against predicted probability of decline.
    Uses bins and
    :param y_true: true labels array.
    :param y_score: scores array.
    :param bins: number of bins.
    :param show: plt.show(). If false, returns axis object.
    """
    ax = hist_of_means(x=y_score, y=y_true, bins=bins, show=False)
    plt.gca().plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='black', lw=3, ls='dashed')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.xlabel('Predicted probability of decline')
    plt.ylabel('True probability of decline')
    plt.title('Calibration')

    if show:
        plt.show()
    else:
        return ax

