import os
from datetime import datetime

import numpy as np
import pandas as pd

from .ClassificationEvaluator import ClassificationEvaluator


class RejectionEvaluator:

    def __init__(
            self,
            filepath: str = None,  # set to None to refrain from loading/recording experiments
            exp_id_cols=None,
            displayed_stats=None,
            overwrite: bool = False) -> object:

        if exp_id_cols is None:
            exp_id_cols = ['predictor', 'rejector']
        if displayed_stats is None:
            displayed_stats = [
                'cov', 'pct_rej_ln', 'pct_rej_lp', '#rej',
                'tpr', 'fpr', 'sel_tpr', 'sel_fpr', 'opt_tpr', 'opt_fpr']

        self.filepath = filepath
        self.exp_id_cols = exp_id_cols
        self.displayed_stats = displayed_stats
        self.overwrite = overwrite

        self.save_results = (self.filepath is not None)
        self.all_stats = [
            'cov', 'pct_rej_ln', 'pct_rej_lp', '#rej',
            'tpr', 'fpr', 'opt_tpr', 'opt_fpr', 'sel_tpr', 'sel_fpr',
            'tn', 'fp', 'fn', 'tp', 'rn', 'rp']
        self.results = pd.DataFrame(columns=(['datetime'] + self.exp_id_cols + self.all_stats))

        if self.save_results:
            if os.path.exists(self.filepath):
                self.results = self.load_csv()

        # using a ClassificationEvaluator for covered test samples
        self.clf_eval = ClassificationEvaluator(
            filepath=None)  # avoids recording results

    def evaluate(self, exp_id: list,
                 y_true: np.ndarray, y_pred: np.ndarray, rejections: np.ndarray,
                 display_results=False):

        counts_cm = self.clf_eval.count_confusion_matrix(
            y_true=y_true[~rejections],
            y_pred=y_pred[~rejections])

        counts_rej = {
            'rn': (rejections & (y_true == 0)).sum(),
            'rp': (rejections & (y_true == 1)).sum()}

        assert (sum(counts_cm.values()) + sum(counts_rej.values())) == y_true.shape[0]

        rej_metrics = self.calc_rej_metrics(**counts_cm, **counts_rej)
        global_metrics = self.calc_global_metrics(**counts_cm, **counts_rej)
        sel_clf_metrics = self.clf_eval.calc_metrics(**counts_cm)
        sel_metrics = {f'sel_{d}': v for d, v in sel_clf_metrics.items()}

        col_names = ['datetime'] + self.exp_id_cols
        values = [datetime.now().strftime("%Y-%m-%d %H:%M")] + exp_id
        for d in [rej_metrics, global_metrics, sel_metrics, counts_cm, counts_rej]:
            for k, v in d.items():
                col_names.append(k)
                values.append(v)

        self.save_to_results(data=values, index=col_names)

        if self.save_results:
            self.save_csv()
        if display_results:
            self.display_results()

    def save_to_results(self, data: list, index: list):

        new_row = pd.Series(data=data, index=index)
        assert list(new_row.index) == list(self.results.columns)

        same_records_ix = (self.results[self.exp_id_cols] == new_row[self.exp_id_cols]).all(axis=1)
        if self.overwrite and same_records_ix.any():
            self.results[same_records_ix] = new_row
        else:
            self.results = self.results.append(new_row, ignore_index=True)

    def get_short_results(self):
        return self.results[self.exp_id_cols + self.displayed_stats]

    def display_results(self, short=False, rej_class=None):
        if short:
            view = self.get_short_results()
        else:
            view = self.results

        if rej_class is not None:
            view = view.loc[view["rej_class"] == rej_class, :]

        print(view)

    def save_csv(self):
        self.results.to_csv(self.filepath, index=False)

    def load_csv(self):
        results = pd.read_csv(self.filepath, index_col=None)
        return results

    @staticmethod
    def calc_rej_metrics(tn, fp, fn, tp, rn, rp, dec_places=4):
        p = fn+tp+rp
        n = tn+fp+rn
        rec_metrics = {
            'cov': 1 - (rn+rp)/(p+n),
            'pct_rej_ln': rn/n,
            'pct_rej_lp': rp/p,
            '#rej': rn+rp
        }
        rec_metrics_rounded = {d: round(v, dec_places) for d, v in rec_metrics.items()}

        return rec_metrics_rounded

    @staticmethod
    def calc_global_metrics(tn, fp, fn, tp, rn, rp, dec_places=4):
        """
        Calculates metrics (e.g. TPR, FPR) allowing for rejection.
        In practice, rejection is taken as a third class.
        """
        p = fn+tp+rp
        n = tn+fp+rn
        rec_metrics = {
            'tpr': tp/p,
            'fpr': fp/n,
            'opt_tpr': (tp+rp)/p,
            'opt_fpr': fp/n
        }

        rec_metrics_rounded = {d: round(v, dec_places) for d, v in rec_metrics.items()}

        return rec_metrics_rounded
