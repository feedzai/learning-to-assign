from autodefer.utils import thresholding as t

from .abstract import AbstractExpert


class MLModelExpert(AbstractExpert):

    def __init__(self, fitted_model, threshold):
        self.model = fitted_model
        self.threshold = threshold

    def predict(self, X, **kwargs):  # kwargs not used (compatibility purposes)
        if self.threshold is None:
            y_pred = self.model.predict_proba(X)[:, 1].squeeze()
        else:
            y_pred = t.predict_with_threshold(
                y_score=self.model.predict_proba(X)[:, self.model.classes_ == 1].squeeze(),
                threshold=self.threshold,
            )

        return y_pred
