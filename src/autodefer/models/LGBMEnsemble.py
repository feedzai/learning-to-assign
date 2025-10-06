import numpy as np

from autodefer.models.run_lgbm import train_lgbm, tune_lgbm_params


class LGBMEnsemble:

    def __init__(self, models_dir=None, n_estimators=10):
        self.models_dir = models_dir
        self.n_estimators = n_estimators

        self.best_individual_params = None
        self.models_list = list()

    def tune_individual_lgbm_params(self, **kwargs):
        """
        Uses tune_lgbm_params to tune the individual parameters of each of the trees in the ensemble.
        Check that function's documentation for argument list.
        """
        self.best_individual_params = tune_lgbm_params(**kwargs)

        return self.best_individual_params

    def fit(
            self,
            logs_path: str,
            X_train: np.ndarray,
            y_train: np.ndarray,
            params: dict,
            n_jobs: int = 1,
            random_seed: int = None,
            deterministic: bool = False):
        """
        Fits self.n_estimator models.
        Each model is itself an ensemble, being a GBT (ensemble).
        """
        for i in range(self.n_estimators):
            model_i_path = f'{self.models_dir}model_{i}.pickle' if self.models_dir is not None else None
            model_i = train_lgbm(
                model_path=model_i_path,
                logs_path=logs_path,
                X_train=X_train,
                y_train=y_train,
                params=params,
                n_jobs=n_jobs,
                random_seed=random_seed,  # updated at the end of the loop
                deterministic=deterministic
            )
            self.models_list.append(model_i)
            random_seed += 1

    def individual_predict_proba_posit(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a (X.shape[0] x self.n_estimators) matrix with the predicted probability score of the positive class.
        """
        scores_list = [  # TODO make efficient
            model_i.predict_proba(X)[:, model_i.classes_ == 1]
            for model_i in self.models_list]

        scores_arr = np.concatenate(scores_list, axis=1)

        return scores_arr  # shape = (X.shape[0] x self.n_estimators)

    def predict_proba_posit(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the mean score from each individual model.
        Use LGBMEnsemble.individual_predict_proba() to access the individual predictions.
        """
        scores_arr = self.individual_predict_proba_posit(X)
        mean_scores = np.mean(scores_arr, axis=1)

        return mean_scores  # shape = (X.shape[0],)

    def total_uncertainty(self, X: np.ndarray):
        """
        Returns total uncertainty
        """
        raise NotImplementedError()
    