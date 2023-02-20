import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sns_scatter_colorbar(
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        hue_label: str = None,
        cmap='rocket_r',
        show=True,
        **kwargs):
    """
    Returns matplotlib object of seaborn scatterplot with a colorbar for hue.
    """
    if hue_label is None:
        hue_label = hue

    norm = plt.Normalize(data[hue].min(), data[hue].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=cmap, **kwargs)
    ax.get_legend().remove()
    cbar = ax.figure.colorbar(sm)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(hue_label, rotation=270)

    if show:
        plt.show()
    else:
        return ax

def hist_of_means(x, y, bins=10, show=True):
    bins_arr = np.linspace(x.min(), x.max() + 1e-12, bins + 1)  # 10 bins, so 11 bin boundaries
    y_bin_id = np.digitize(x, bins_arr)
    ax = plt.bar(bins_arr[:-1], [np.mean(y[y_bin_id == i]) for i in range(1, len(bins_arr))],
                 width=bins_arr[1] - bins_arr[0], align='edge', fc='steelblue', ec='black')
    # plt.xticks(bins)
    plt.margins(x=0.02)  # smaller margins

    if show:
        plt.show()
    else:
        return ax
