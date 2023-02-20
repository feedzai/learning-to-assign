from ..tuner import OptunaTuner
from .visualization import plot_search

class BinaryClassTuner(OptunaTuner):
    def __init__(self, *args, **kwargs):
        super().__init__(task='binary', *args, **kwargs)

    def plot_search(self, *args, **kwargs):
        return plot_search(
            y_true=self.y_val,
            y_scores=self.preds_hist,
            params=self.params_hist,
            *args, **kwargs
        )
