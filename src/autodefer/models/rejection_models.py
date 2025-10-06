import numpy as np

from autodefer.utils import thresholding as t


class RejectionAfterFPRThresholding:

    def __init__(self):
        self.t_negat_rejector = None
        self.t_posit_base = None

    def threshold(
            self,
            y_true: np.ndarray,
            classifier_score: np.ndarray,
            rejector_score: np.ndarray,
            coverage: float,
            fpr: float):
        """
        Sets self.t_negat_uncertainty and self.t_posit_base.
        Thresholding performed in the provided y_true (be it a validation or test set).
        self.t_posit_base is set first to ensure the specified FPR.
        self.negat_uncertainty is dependent on t_posit_base and ensures the specified coverage.
        """
        self.t_posit_base = t.calc_threshold_at_fpr(
            y_true=y_true,
            y_score=classifier_score,
            fpr=fpr
        )
        non_pred_posit = (classifier_score < self.t_posit_base)
        n_rejected = int((1-coverage)*y_true.shape[0])
        sorted_filtered_uncertainty_score = np.sort(rejector_score[non_pred_posit])  # ascending
        self.t_negat_rejector = sorted_filtered_uncertainty_score[-n_rejected]  # TODO make robust to uniform scores

    def predict_and_reject(
            self,
            classifier_score: np.ndarray,
            rejector_score: np.ndarray):
        """
        Returns tuple with predictions and rejections.
        Dependent on self.t_negat_uncertainty and self.t_posit_base.
        Thresholds must be set beforehand using self.threshold().
        """
        pred_posit_ix = classifier_score >= self.t_posit_base
        weak_pred_negat_ix = rejector_score <= self.t_negat_rejector  # weak because pred_posit_ix overrules

        y_pred = np.full(shape=rejector_score.shape, fill_value=-1, dtype=int)
        y_pred[weak_pred_negat_ix] = 0  # maintains dtype=int
        y_pred[pred_posit_ix] = 1

        y_rej = (y_pred == -1)  # dtype=bool

        return y_pred, y_rej
