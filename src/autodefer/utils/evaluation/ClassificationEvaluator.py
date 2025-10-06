import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics


class ClassificationEvaluator:

    def __init__(
            self,
            filepath: str = None,  # set to None to avoid loading/recording records
            exp_id_cols=None,
            displayed_stats=None,
            overwrite: bool = False) -> object:

        if exp_id_cols is None:
            exp_id_cols = ['model']
        if displayed_stats is None:
            displayed_stats = ['tpr', 'fpr']

        self.filepath = filepath
        self.exp_id_cols = exp_id_cols
        self.displayed_stats = displayed_stats
        self.overwrite = overwrite

        self.save_results = (self.filepath is not None)
        self.all_stats = [
            'acc', 'tpr', 'precision', 'fpr', '%lp', '%ln', '%pp', '%pn',
            'tn', 'fp', 'fn', 'tp']
        self.results = pd.DataFrame(columns=(['datetime'] + self.exp_id_cols + self.all_stats))

        if self.save_results:
            if os.path.exists(self.filepath):
                self.results = self.load_csv()

    def evaluate(
            self,
            exp_id: list,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            display_results=False):
        if isinstance(exp_id, tuple):
            exp_id = list(exp_id)
        elif isinstance(exp_id, list):
            pass
        else:
            raise ValueError('exp_id must be a list or a tuple.')

        counts_cm = self.count_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred)

        rec_metrics = self.calc_metrics(**counts_cm)

        col_names = ['datetime'] + self.exp_id_cols
        values = [datetime.now().strftime("%Y-%m-%d %H:%M")] + exp_id
        for d in [rec_metrics, counts_cm]:
            for k, v in d.items():
                col_names.append(k)
                values.append(v)

        self.save_to_results(data=values, index=col_names)

        if self.save_results:
            self.save_csv()
        if display_results:
            self.display_results()

    def evaluate_df(self, y_true: np.ndarray, y_preds: pd.DataFrame, display_results=False):
        for exp_id, y_pred in y_preds.iteritems():
            if not isinstance(exp_id, (list, tuple)):
                exp_id = [exp_id]
            self.evaluate(exp_id=exp_id, y_true=y_true, y_pred=y_pred, display_results=display_results)

    def save_to_results(self, data: list, index: list):

        new_row = pd.Series(data=data, index=index)
        assert list(new_row.index) == list(self.results.columns)

        same_records_ix = (self.results[self.exp_id_cols] == new_row[self.exp_id_cols]).all(axis=1)
        if self.overwrite and same_records_ix.any():
            self.results[same_records_ix] = new_row
        else:
            self.results = self.results.append(new_row, ignore_index=True)

    def get_results(self, short=False):
        if short:
            results = self.results[self.exp_id_cols + self.displayed_stats]
        else:
            results = self.results

        return results

    def display_results(self, short=False):
        print(self.get_results(short=short))

    def save_csv(self):
        self.results.to_csv(self.filepath, index=False)

    def load_csv(self):
        results = pd.read_csv(self.filepath, index_col=None)
        return results

    @staticmethod
    def calc_metrics(tn, fp, fn, tp, dec_places=4) -> dict:
        rec_metrics = {
            'acc': (tn + tp) / (tn + fp + fn + tp),
            'tpr': tp / (tp + fn),
            'precision': tp / (tp + fp),
            'fpr': fp / (tn + fp),
            '%lp': (tp + fn) / (tn + fp + fn + tp),
            '%ln': (tn + fp) / (tn + fp + fn + tp),
            '%pp': (tp + fp) / (tn + fp + fn + tp),
            '%pn': (tn + fn) / (tn + fp + fn + tp),
        }

        rec_metrics_rounded = {d: round(v, dec_places) for d, v in rec_metrics.items()}

        return rec_metrics_rounded

    @staticmethod
    def count_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=[0, 1]).ravel()

        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
