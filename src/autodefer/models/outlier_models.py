import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from autodefer.external_files.trustscore import TrustScore


class ScaledLocalOutlierFactor:

    def __init__(self, lof_model_path: str = None, **kwargs):
        self.lof_model_path = lof_model_path

        self.ss = StandardScaler()
        # load trust score model if saved
        if os.path.exists(self.lof_model_path):
            self.loaded_model = True
            with open(self.lof_model_path, 'rb') as infile:
                self.lof_model = pickle.load(infile)
            print(f'Model loaded from {self.lof_model_path}')
        else:
            self.loaded_model = False
            self.lof_model = LocalOutlierFactor(**kwargs)

    def fit(self, X_train, logs_path: str = None):
        """
        Fits both the standard scaler and the trust score.
        """
        # standard scaler
        self.ss.fit(X_train)
        X_train_std = self.ss.transform(X_train)

        if self.loaded_model:
            print(f'Model has already been fitted and loaded from {self.lof_model_path}')
            return None

        # trust score
        start_time = datetime.now()
        self.lof_model.fit(X_train_std)
        finish_time = datetime.now()

        if self.lof_model_path is not None:
            with open(self.lof_model_path, 'wb') as outfile:
                pickle.dump(self.lof_model, outfile)

        # record training time
        duration = finish_time - start_time
        log_msg = (
                f'Training at {self.lof_model_path}: ' +
                f'finish_time={str(finish_time)}, ' +
                f'duration=({str(duration)})'
        )
        print(log_msg)
        if logs_path is not None:
            with open(logs_path, 'a') as outfile:
                outfile.write(log_msg)

    def score_samples(
            self,
            X: np.ndarray,
            scores_path: str = None):

        if os.path.exists(scores_path):
            lof_score = np.load(scores_path)
            print(f'Scores loaded from {scores_path}')
            return lof_score

        X_std = self.ss.transform(X)
        lof_score = self.lof_model.score_samples(X=X_std)

        if scores_path is not None:
            np.save(scores_path[:-4], lof_score)  # [:-4] to remove .npy
            print(f'Scores saved to {scores_path}')

        return lof_score


class ScaledTrustScore:

    def __init__(self, trust_score_model_path: str = None):
        self.trust_score_model_path = trust_score_model_path

        self.ss = StandardScaler()
        # load trust score model if saved
        if os.path.exists(self.trust_score_model_path):
            self.loaded_model = True
            with open(self.trust_score_model_path, 'rb') as infile:
                self.trust_score_model = pickle.load(infile)
        else:
            self.loaded_model = False
            self.trust_score_model = TrustScore()

    def fit(self, X_train, y_train, logs_path: str = None):
        """
        Fits both the standard scaler and the trust score.
        """
        # standard scaler
        self.ss.fit(X_train)
        X_train_std = self.ss.transform(X_train)

        if self.loaded_model:
            print(f'Model has already been fitted and loaded from {self.trust_score_model_path}')
            return None

        # trust score
        start_time = datetime.now()
        self.trust_score_model.fit(X_train_std, y_train)
        finish_time = datetime.now()

        if self.trust_score_model_path is not None:
            with open(self.trust_score_model_path, 'wb') as outfile:
                pickle.dump(self.trust_score_model, outfile)

        # record training time
        duration = finish_time - start_time
        log_msg = (
                f'Training at {self.trust_score_model_path}: ' +
                f'finish_time={str(finish_time)}, ' +
                f'duration=({str(duration)})'
        )
        print(log_msg)
        if logs_path is not None:
            with open(logs_path, 'a') as outfile:
                outfile.write(log_msg)

    def score_samples(
            self,
            X: np.ndarray,
            y_pred: np.ndarray,
            scores_path: str = None):

        if os.path.exists(scores_path):
            trust_score = np.load(scores_path)
            print(f'Scores loaded from {scores_path}')
            return trust_score

        X_std = self.ss.transform(X)
        trust_score_raw = self.trust_score_model.get_score(X=X_std, y_pred=y_pred)
        trust_score = trust_score_raw.astype(float)  # dtype=object originally

        if scores_path is not None:
            np.save(scores_path[:-4], trust_score)  # [:-4] to remove .npy

        return trust_score


def train_isolation_forest(
        X_train: np.ndarray,
        params: dict,
        n_jobs: int = 1,
        random_seed: int = None,
        model_path: str = None,  # loads from if saved; saves to otherwise
        logs_path: str = None  # registers running times in
        ):
    # load model pickle and return it if it has already been saved to model_path
    if os.path.exists(model_path):
        with open(model_path, 'rb') as infile:
            model = pickle.load(infile)
        print(f'Model loaded from {model_path}')
        return model

    start_time = datetime.now()
    params.update({  # "operational" parameters must be set
        'n_jobs': n_jobs,
        'random_state': random_seed
    })
    model = IsolationForest(**params)
    model.fit(X_train)
    finish_time = datetime.now()

    if model_path is not None:
        with open(model_path, 'wb') as outfile:
            pickle.dump(model, outfile)

    # record training time
    duration = finish_time - start_time
    log_msg = (
            f'Training at {model_path}: ' +
            f'finish_time={str(finish_time)}, ' +
            f'duration=({str(duration)})'
    )
    print(log_msg)
    with open(logs_path, 'a') as outfile:
        outfile.write(log_msg)

    return model


def score_with_isolation_forest(
        model,
        X: np.ndarray,
        scores_path: str = None
        ):

    if os.path.exists(scores_path):
        iso_forest_score = np.load(scores_path)
        return iso_forest_score

    iso_forest_score = model.score_samples(X)

    if scores_path is not None:
        np.save(scores_path[:-4], iso_forest_score)  # [:-4] to remove .npy
        print(f'Scores saved to {scores_path}')

    return iso_forest_score
