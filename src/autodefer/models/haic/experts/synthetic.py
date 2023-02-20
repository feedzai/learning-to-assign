import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .generate_features import randomly_invert_labels, invert_labels_with_probabilities

from .abstract import AbstractExpert


class ArbitrarilyAccurateExpert(AbstractExpert):

    def __init__(self, accuracy: float, seed=None):
        self.accuracy = accuracy
        self.seed = seed if seed is not None else np.random.default_rng().integers(low=0, high=2 ** 32 - 1)

    def predict(self, y, **kwargs):  # kwargs not used (compatibility purposes)
        decisions = randomly_invert_labels(
            labels=y, p=(1 - self.accuracy), seed=self.seed
        )

        return decisions


class LinearlyAccurateExpert(AbstractExpert):

    def __init__(self, accuracy: float, seed=None):
        raise NotImplementedError  # TODO not up to date with LinearlyAccurateBinaryExpert
        self.accuracy = accuracy
        self.seed = seed if seed is not None else np.random.default_rng().integers(low=0, high=2 ** 32 - 1)

        # params set by fit
        self.ohe = None
        self.categorical_cols = None
        self.scaler = None

        self.beta_zero = None
        self.betas = None

    def fit(self, X, categorical_cols=None):
        self.categorical_cols = categorical_cols
        X = self._one_hot_encode(X)
        X = self._normalize(X)

        rng = np.random.default_rng(self.seed)
        self.beta_zero = 1 - self.accuracy
        target_st_dev = 0.05  # standard deviation of the probability of error
        self.betas = (
            rng.normal(
                loc=0,
                scale=target_st_dev / ((X.shape[1]) ** 0.5),
                size=(X.shape[1],),
            )
        )

    def predict(self, X, y, **kwargs):  # kwargs not used (compatibility purposes)
        if self.betas is None:
            raise ValueError('Synthetic expert must be .fit() to the data.')

        X = self._one_hot_encode(X)
        X = self._normalize(X)

        probability_of_error = np.clip(
            (X * self.betas).sum(axis=1) + self.beta_zero,
            a_min=0,
            a_max=1,
        )
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.histplot(probability_of_error)
        # plt.show()  # TODO

        decisions = invert_labels_with_probabilities(
            labels_arr=y,
            p_arr=probability_of_error,
            seed=self.seed
        )

        return decisions

    def _one_hot_encode(self, X):
        if self.categorical_cols is None:
            return X

        if self.ohe is None:
            self.ohe = OneHotEncoder(drop='first')  # will be used by linear regression
            self.ohe.fit(X.loc[:, self.categorical_cols])

        ohe_ft_arr = self.ohe.transform(X.loc[:, self.categorical_cols]).toarray()
        ohe_ft_df = pd.DataFrame(
            ohe_ft_arr,
            index=X.index,
            columns=self.ohe.get_feature_names(self.categorical_cols)  # update to .get_feature_names_out() after v1.0
        )
        X_with_ohe = pd.concat(objs=(X, ohe_ft_df), axis=1).drop(columns=self.categorical_cols)

        return X_with_ohe

    def _normalize(self, X):
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        X_scaled = self.scaler.transform(X)

        return X_scaled


class LinearlyAccurateBinaryExpert(AbstractExpert):

    def __init__(
            self,
            fnr: float, fpr: float,
            fnr_beta_score: float, fpr_beta_score: float,
            fnr_beta_protected: float, fpr_beta_protected: float,
            fnr_betas_stdev: float,
            fpr_betas_stdev: float,
            fnr_betas_min: float,
            fpr_betas_min: float,
            fnr_noise_stdev: float = 0,
            fpr_noise_stdev: float = 0,
            seed=None,
    ):
        self.fnr = fnr
        self.fpr = fpr
        self.fnr_beta_score = fnr_beta_score
        self.fpr_beta_score = fpr_beta_score
        self.fnr_beta_protected = fnr_beta_protected
        self.fpr_beta_protected = fpr_beta_protected

        self.fnr_betas_stdev = fnr_betas_stdev
        self.fpr_betas_stdev = fpr_betas_stdev
        self.fnr_betas_min = fnr_betas_min
        self.fpr_betas_min = fpr_betas_min
        self.fnr_noise_stdev = fnr_noise_stdev
        self.fpr_noise_stdev = fpr_noise_stdev

        self.seed = (
            seed if seed is not None
            else np.random.default_rng().integers(low=0, high=2 ** 32 - 1)
        )

        # params set by fit
        self.fnr_intercept = None
        self.fnr_betas = None
        self.fpr_intercept = None
        self.fpr_betas = None

    def fit(self, X, y, score_col, protected_col):
        fixed_fnr_betas = {
            X.columns.get_loc(score_col): self.fnr_beta_score,
            X.columns.get_loc(protected_col): self.fnr_beta_protected,
        }
        fixed_fpr_betas = {
            X.columns.get_loc(score_col): self.fpr_beta_score,
            X.columns.get_loc(protected_col): self.fpr_beta_protected,
        }

        # FALSE NEGATIVES
        self.fnr_intercept = self.fnr
        rng = np.random.default_rng(self.seed)
        self.fnr_betas = rng.normal(loc=0, scale=self.fnr_betas_stdev, size=(X.shape[1],))
        self.fnr_betas[self.fnr_betas < self.fnr_betas_min] = 0
        for fixed_beta_ix, fixed_beta_value, in fixed_fnr_betas.items():
            self.fnr_betas[fixed_beta_ix] = fixed_beta_value

        # FALSE POSITIVES
        self.fpr_intercept = self.fpr
        self.fpr_betas = rng.normal(loc=0, scale=self.fpr_betas_stdev, size=(X.shape[1],))
        self.fpr_betas[self.fpr_betas < self.fpr_betas_min] = 0
        for fixed_beta_ix, fixed_beta_value, in fixed_fpr_betas.items():
            self.fpr_betas[fixed_beta_ix] = fixed_beta_value

        # ADJUST INTERCEPTS ACCORDING TO EMPIRICAL PERFORMANCE
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            y_true=y,
            y_pred=self.predict(X=X, y=y),
            labels=[0, 1]
        ).ravel()
        empirical_fnr = fn/(tp+fn)
        empirical_fpr = fp/(tn+fp)
        self.fnr_intercept += (self.fnr - empirical_fnr)
        self.fpr_intercept += (self.fpr - empirical_fpr)

    def predict(self, X, y, **kwargs):  # kwargs not used (compatibility purposes)
        if self.fnr_betas is None:
            raise ValueError('Synthetic expert must be .fit() to the data.')

        np_rng = np.random.default_rng(self.seed)

        probability_of_fn = (y == 1) * (np.clip(
            self.fnr_intercept + (X * self.fnr_betas).sum(axis=1)
            + np_rng.normal(loc=0, scale=self.fnr_noise_stdev, size=(X.shape[0],)),
            a_min=0,
            a_max=1,
        ))
        probability_of_fp = (y == 0) * np.clip(
            self.fpr_intercept + (X * self.fpr_betas).sum(axis=1)
            + np_rng.normal(loc=0, scale=self.fpr_noise_stdev, size=(X.shape[0],)),
            a_min=0,
            a_max=1,
        )
        probability_of_error = probability_of_fn + probability_of_fp

        decisions = invert_labels_with_probabilities(
            labels_arr=y,
            p_arr=probability_of_error,
            seed=self.seed
        )

        return decisions
