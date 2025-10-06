from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd

from .assignment_functions import random_assignment

class AbstractAssigner(ABC):

    def __init__(self, expert_ids):
        if isinstance(expert_ids, dict):
            self.expert_ids = self.flatten_dict_of_lists(expert_ids)
        else:
            self.expert_ids = expert_ids

    @abstractmethod
    def assign(self, X):
        raise NotImplementedError()

    @staticmethod
    def flatten_dict_of_lists(dict_of_lists):
        return [
            item
            for sublist in [dict_of_lists[_type] for _type in dict_of_lists]
            for item in sublist
        ]

class RandomAssigner(AbstractAssigner):

    def __init__(self, expert_ids, seed=None):
        super().__init__(expert_ids)
        self.seed = seed if seed is not None else np.random.default_rng().integers(low=0, high=2**32-1)

    def assign(self, X, batch_col=None, capacity=None):
        if batch_col is None:
            batch_col = 'batch'
            X[batch_col] = 0
            if capacity is None:
                capacity = {
                    0: {expert_id: self._get_equal_capacity(n=X.shape[0])
                        for expert_id in self.expert_ids}
                }
        else:
            if capacity is None:
                raise ValueError('If batch_col is None, capacity must be a dict matching testbed to capacity dicts.')

        assignments = list()
        for b in X[batch_col].unique():
            batch_X = X[X[batch_col] == b].drop(columns=batch_col)
            batch_capacity = capacity[b]
            assignments.append(random_assignment(
                X=batch_X, capacity=batch_capacity, random_seed=self.seed,
            ))

        assignments = pd.concat(assignments)

        return assignments

    def _get_equal_capacity(self, n):
        capacity = {expert_id: int(n/len(self.expert_ids)) for expert_id in self.expert_ids}

        # correct rounding down errors
        np_rng = np.random.default_rng(seed=self.seed)
        shuffled_expert_ids = deepcopy(self.expert_ids)
        np_rng.shuffle(shuffled_expert_ids)
        for expert_id in shuffled_expert_ids:
            if sum(capacity.values() == n):
                break
            capacity[expert_id] += 1
