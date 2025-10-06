import numpy as np
import pandas as pd


class ExpertTeam(dict):

    def __init__(self, experts=None):
        if experts is None:
            experts = dict()
        self.experts = self._convert_to_dict(experts)
        super().__init__(experts)

    def fit(self, **kwargs):
        for _, expert_obj in self.items():
            try:
                expert_obj.fit(**kwargs)
            except AttributeError:  # experts that do not need to fit
                pass

    def predict(
            self,
            index,
            predict_kwargs: dict,
            long_format=False, assignment_col=None, decision_col=None,
    ):
        predictions_dict = dict()
        for expert_id, expert in self.items():
            predictions_dict[expert_id] = expert.predict(**predict_kwargs[type(expert)])
        predictions_df = pd.DataFrame(predictions_dict, index=index, columns=list(self.keys()))

        if long_format:
            predictions_df = predictions_df.reset_index()
            predictions_df = predictions_df.melt(
                id_vars=index.name,
                var_name=assignment_col,
                value_name=decision_col
            )

        return predictions_df

    def query(self, index, assignments, **kwargs):
        predictions = self.predict(index, **kwargs)
        mask = np.array(
            [assignments == e for e in predictions.columns]
        ).T
        queried_decisions = pd.Series(
            predictions.values[mask],
            index=index
        )

        return queried_decisions

    @staticmethod
    def _convert_to_dict(experts) -> dict:
        if isinstance(experts, (list, tuple)):
            experts_dict = {i: experts[i] for i in range(len(experts))}
        elif isinstance(experts, dict):
            experts_dict = experts
        else:
            raise ValueError('experts must be either a list, a tuple, or, preferibly, a dict.')

        return experts_dict
