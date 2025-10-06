import numpy as np


class PDR:
    """
    Class to aid calculation of percent dollar regret (PDR).
    Source: "Real Tasks, Real Data, Real Users: Evaluating the Efficacy of Post-hoc Explainable Machine Learning
    Methods"
    Parameter defaults in accordance with the source.
    """

    def __init__(self, alpha=-3, beta=0.5, delta=0.1, lambda_=3):
        """
        :param lambda_: true negative revenue multiplier (long-term)
        :param alpha: false negative cost multiplier (long-term)
        :param beta: probability of losing the current sale
        :param delta: probability of losing long-term revenue from false positive
        :return: PDR
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.lambda_ = lambda_

    def calc_realized_revenue(self, labels, decisions, values):
        """
        Calculates the realized revenue from each decision.
        :param labels: numpy array (or pandas series) of ground truth labels
        :param decisions: numpy array (or pandas series) of decisions (predictions)
        :param values: numpy array (or pandas series) of transaction values
        :return: numpy array (or pandas series) of the unrealized revenue
        """
        true_positve_perc_rev = (
            ((labels == 0) & (decisions == 0)) *
            (1 + self.lambda_)
        )
        false_positives_perc_rev = (
            ((labels == 0) & (decisions == 1)) *
            ((1 - self.beta) + (1 - self.delta) * self.lambda_)
        )
        false_negative_perc_rev = (
            ((labels == 1) & (decisions == 0)) *
            self.alpha  # alpha < 0
        )
        total_unrealized_rev = (true_positve_perc_rev + false_positives_perc_rev + false_negative_perc_rev) * values

        return total_unrealized_rev

    def calc_unrealized_revenue(self, labels, decisions, values):
        """
        Calculates the unrealized revenue from each decision.
        :param labels: numpy array (or pandas series) of ground truth labels
        :param decisions: numpy array (or pandas series) of decisions (predictions)
        :param values: numpy array (or pandas series) of transaction values
        :return: numpy array (or pandas series) of the unrealized revenue
        """
        false_positives_perc_cost = (
            ((labels == 0) & (decisions == 1)) *
            (self.beta + self.delta * self.lambda_)
        )
        false_negative_perc_cost = (
            ((labels == 1) & (decisions == 0)) *
            (-self.alpha)
        )
        total_unrealized_rev = (false_positives_perc_cost + false_negative_perc_cost) * values

        return total_unrealized_rev

    def calc_possible_revenue(self, labels, values):
        """
        Calculates the possible revenue from each case.
        :param labels: numpy array (or pandas series) of ground truth labels
        :param values: numpy array (or pandas series) of transaction values
        :return: numpy array (or pandas series) of the possible revenue
        """
        return (labels == 0) * (1 + self.lambda_) * values

    def calc_pdr(self, labels, decisions, values):
        """
        Calculates the percent dollar regret from all decisions.
        :param labels: numpy array (or pandas series) of ground truth labels
        :param decisions: numpy array (or pandas series) of decisions (predictions)
        :param values: numpy array (or pandas series) of transaction values
        :return: float of the percent dollar regret
        """
        unrealized_rev = self.calc_unrealized_revenue(labels=labels, decisions=decisions, values=values)
        possible_rev = self.calc_possible_revenue(labels=labels, values=values)

        return self.calc_pdr_from_revenues(unrealized_revenue=unrealized_rev, possible_revenue=possible_rev)

    def calc_individual_pdr(self, labels, decisions, values):
        """
        Calculates the percent dollar regret from each individual decision.
        :param labels: numpy array (or pandas series) of ground truth labels
        :param decisions: numpy array (or pandas series) of decisions (predictions)
        :param values: numpy array (or pandas series) of transaction values
        :return: numpy array (or pandas series) of the percent dollar regret
        """
        unrealized_rev = self.calc_unrealized_revenue(labels=labels, decisions=decisions, values=values)
        possible_rev = self.calc_possible_revenue(labels=labels, values=values)
        individual_pdr = np.divide(
            unrealized_rev, possible_rev,
            out=np.full_like(unrealized_rev, fill_value=np.nan),
            where=(possible_rev != 0)  # ensures NaN when divided by 0
        )

        return individual_pdr

    @staticmethod
    def calc_pdr_from_revenues(unrealized_revenue, possible_revenue):
        """

        :param unrealized_revenue: numpy array (or pandas series)
        :param possible_revenue: numpy array (or pandas series)
        :return: float of the percent dollar regret
        """

        total_possible_rev = possible_revenue.sum()
        if total_possible_rev != 0:
            pdr_ = unrealized_revenue.sum() / total_possible_rev
        else:
            pdr_ = None

        return pdr_
