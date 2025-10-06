import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .metrics import SUPPORTED_METRICS, calc_metrics

def plot_search(
        y_true, y_scores: list, params: list,
        x: str, y: str,
        hue: str = None, size: str = None,
        title: str = None,
        show=True,
):
    ax = sns.scatterplot(
        x=_get_values(criterion=x, y_true=y_true, y_scores=y_scores, params=params),
        y=_get_values(criterion=y, y_true=y_true, y_scores=y_scores, params=params),
        hue=_get_values(criterion=hue, y_true=y_true, y_scores=y_scores, params=params),
        size=_get_values(criterion=size, y_true=y_true, y_scores=y_scores, params=params),
    )
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    if show:
        plt.show()
    else:
        return ax

# aux functions
def _get_values(criterion, y_true, y_scores, params: list) -> list:
    if criterion in SUPPORTED_METRICS:
        values = calc_metrics(metric_name=criterion, y_true=y_true, y_scores=y_scores)

    else:
        if criterion == 'algorithm':
            values = np.array(
                [params[i]['classpath'].split('.')[-1] for i in range(len(params))]
            )
        elif criterion == 'trial':
            values = np.array([i+1 for i in range(len(params))])
        elif criterion is None:
            values = None
        else:
            raise ValueError(f'Criterion "{criterion}" not supported.')

    return values
