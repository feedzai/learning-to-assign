import numpy as np
import pandas as pd

from autodefer.utils.evaluation import ClassificationEvaluator

def query_experts(pred, assignments):
    assert pred.shape[0] == assignments.shape[0]
    mask = np.array([assignments == e for e in pred.columns]).T
    queried_decisions = pd.Series(pred.values[mask], index=pred.index)

    return queried_decisions

class HAICEvaluator:

    def __init__(self, y_true, experts_pred, exp_id_cols, displayed_stats=None):
        self.y_true = y_true
        self.experts_pred = experts_pred
        self.clf_eval = ClassificationEvaluator(
            exp_id_cols=exp_id_cols,
            displayed_stats=displayed_stats,
            filepath=None
        )

        self.assignments = dict()
        self.decisions = dict()

    def evaluate(
            self,
            exp_id,
            assignments,
            decisions=None,
            batches=None,
            capacity=None,
            assert_capacity_constraints=True,
    ):
        if assert_capacity_constraints:
            self._assert_capacity_constraints(assignments, batches, capacity)

        if decisions is None:
            decisions = query_experts(pred=self.experts_pred, assignments=assignments)

        self.clf_eval.evaluate(
            exp_id=exp_id,
            y_true=self.y_true,
            y_pred=decisions,
        )

        # record arrays for external use
        self.assignments[exp_id] = assignments
        self.decisions[exp_id] = decisions

    def get_results(self, *args, **kwargs):
        return self.clf_eval.get_results(*args, **kwargs)

    @staticmethod
    def _assert_capacity_constraints(assignments, batches, capacity):
        if batches is None:
            batches = pd.DataFrame(
                np.full(shape=(assignments.shape[0],), fill_value=0),
                index=assignments.index
            )

        if capacity is not None:
            for b in batches.iloc[:, 0].unique():
                batch_ix = assignments.index[(batches.iloc[:, 0] == b)]
                batch_capacity = capacity[b]
                for expert_id in batch_capacity:
                    if (assignments[batch_ix] == expert_id).sum() != batch_capacity[expert_id]:
                        raise ValueError('Assignments do not respect capacity constraints')
