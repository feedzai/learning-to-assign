import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection


class SelectiveLabelsEvaluator:

    def __init__(
            self, data: pd.DataFrame,
            human_id_col: str, decision_col: str,
            scores_col: str, false_negative_col: str,
            protected_attribute_col: str = None
    ):
        self.data = data
        self.human_id_col = human_id_col
        self.decision_col = decision_col
        self.false_negative_col = false_negative_col
        self.scores_col = scores_col
        self.protected_attribute_col = protected_attribute_col

        self.FAIRNESS_AWARE = (self.protected_attribute_col is not None)

        self.human_stats = self.calc_human_performance_stats()
        self.contractions = None

        self.set_contractions()

    def calc_contractions(self):
        edited_data = (  # data to use for calculations
            self.data[
                [self.human_id_col, self.decision_col, self.false_negative_col, self.scores_col] +
                ([self.protected_attribute_col] if self.FAIRNESS_AWARE else [])
            ]
            .rename(columns={
                **{
                    self.human_id_col: 'human_id',
                    self.decision_col: 'decision',
                    self.false_negative_col: 'false_negative',
                    self.scores_col: 'score',
                },
                **({self.protected_attribute_col: 'protected_attribute'} if self.FAIRNESS_AWARE else {})
             })
        )

        human_ids = self.get_humans()
        contraction_dfs = list()
        for human_id in human_ids:
            h_data = edited_data[edited_data['human_id'] == human_id]

            # contraction score equals to the model score, unless it is a decline (must also be decline, then)
            h_data['contraction_score'] = h_data['score']
            h_data['contraction_score'][h_data['decision'] == 1] = 1

            # accumulating statistics as decline window expands
            h_data = (
                h_data
                .sort_values(by='contraction_score', ascending=False)
                .reset_index(drop=True)
            )

            h_data['decline_rate'] = (h_data.index + 1) / h_data.shape[0]
            # h_data['decline_rate'][h_data['decision'] == 1] = None

            h_data['contraction_false_neg_count'] = (
                h_data['false_negative'].sum() - h_data['false_negative'].cumsum()
            )
            h_data['contraction_false_neg'] = h_data['contraction_false_neg_count'] / h_data.shape[0]

            if self.FAIRNESS_AWARE:
                # demographic parity (goal is to compare between humans/AI)
                h_data['contraction_fairness_ratio_log2'] = np.log2(
                    (
                        (1 - h_data['protected_attribute']).expanding().sum() /
                        (1 - h_data['protected_attribute']).sum()
                    ) /
                    (
                        h_data['protected_attribute'].expanding().sum() /
                        h_data['protected_attribute'].sum()
                    )
                )

            h_data['decline_agreement'] = (
                    h_data.sort_values(by='score', ascending=False)['decision'].cumsum().values /
                    (h_data['decision'] == 1).sum()
            )

            h_data = h_data.iloc[  # everything before is not contraction, nor the first point
                h_data[h_data['decision'] == 1].index[-1]:,
            ]

            contraction_dfs.append(h_data.reset_index(drop=True))

        contractions = pd.concat(contraction_dfs, axis=0).reset_index(drop=True)

        return contractions

    def calc_human_performance_stats(self):
        def get_stats(x, scores_col, decision_col, false_negative_col, protected_attribute_col):
            stats = {
                'volume': x.shape[0],
                'avg_score': x[scores_col].mean(),
                'decline_rate': x[decision_col].mean(),
                'declines': x[decision_col].sum(),
                'false_neg': x[false_negative_col].mean(),
                'false_negs': x[false_negative_col].sum(),
            }

            if self.FAIRNESS_AWARE:
                decline_conditional_prob = {}
                for i in [0, 1]:
                    decline_conditional_prob[i] = (
                        (
                            (x[protected_attribute_col] == i) &
                            (x[decision_col] == 1)
                        ).sum() /
                        (x[protected_attribute_col] == i).sum()
                    )
                stats['fairness_ratio_log2'] = np.log2(
                    decline_conditional_prob[0] / decline_conditional_prob[1] if decline_conditional_prob[1] != 0
                    else float('inf')
                )

            return pd.Series(stats.values(), index=stats.keys())

        human_stats = (
            self.data
            .groupby(self.human_id_col)
            .apply(
                lambda x: get_stats(
                    x,
                    scores_col=self.scores_col, decision_col=self.decision_col,
                    false_negative_col=self.false_negative_col, protected_attribute_col=self.protected_attribute_col)
            )
            .reset_index(drop=False)
            .rename(columns={self.human_id_col: 'human_id'})
        )

        return human_stats

    # viz --------------------------------------------------------------------------------------------------------------
    def plot_contractions(
            self,
            evaluatee_ids=None,
            lenient_ids=None,
            y: str = 'false_neg',
            xlim=(0, 1),
            ylim=(None, None),
            title='Model contractions (lines) vs analysts (crosses)',
            line_hue='volume',
            line_cmap=None,
            line_color='steelblue',
            line_alpha=0.6,
            line_width=0.65,
            scatter_color='orange',
            scatter_marker='X',
            scatter_size=50,
            lineplot_kwargs: dict = None,
            scatterplot_kwargs: dict = None,
            show: bool = True
    ):
        """
        Plots false negative prevalence vs. decline rate across humans and contractions.
        :param evaluatee_ids: list of IDs to filter considered evaluatees.
        :param lenient_ids: list of IDs to filter considered lenients.
        :param y: 'false_neg' or 'fairness_ratio_log2'.
        :param xlim: unpacked in plt.xlim().
        :param ylim: unpacked in plt.ylim().
        :param title: fed to plt.title().
        :param line_hue: 'volume' and 'decline_agreement'.
        :param line_cmap: cmap for lineplot.
        :param line_color: color to be used if line_hue is None.
        :param line_alpha: alpha in lineplot.
        :param line_width: lw in lineplot.
        :param scatter_color: color in sns.scatterplot().
        :param scatter_marker: marker in sns.scatterplot().
        :param scatter_size: s in sns.scatterplot().
        :param lineplot_kwargs: kwargs fed to lineplot.
        :param scatterplot_kwargs: kwargs fed to sns.scatterplot().
        :param show: whether to plt.show() the plot, or to return it.
        :return: matplotlib plot.
        """
        if evaluatee_ids is None:
            evaluatee_ids = self.get_humans()
        if lenient_ids is None:
            lenient_ids = self.get_humans()

        if lineplot_kwargs is None:
            lineplot_kwargs = dict()
        if scatterplot_kwargs is None:
            scatterplot_kwargs = dict()

        fig, ax = plt.subplots()
        if line_hue == 'volume' or line_hue is None:  # if None, we still need this to separate lines
            lineplot_data = self.contractions.merge(
                self.human_stats.add_prefix('lenient_'),
                left_on='human_id', right_on='lenient_human_id')
            # sns.lineplot(hue) groups the lines. As such, volume must be unique.
            unique_jitter = dict(zip(
                lenient_ids,
                list(range(len(lenient_ids)))
            ))
            lineplot_data['jittered_volume'] = (
                    lineplot_data['lenient_volume'] +
                    0.001 * lineplot_data['human_id'].map(unique_jitter)
            )
            sns.lineplot(
                data=lineplot_data[lineplot_data['human_id'].isin(lenient_ids)],
                x='decline_rate', y=f'contraction_{y}',
                zorder=-1,
                hue='jittered_volume',
                color=line_color if line_hue is None else None,
                palette=sns.cubehelix_palette(as_cmap=True) if line_cmap is None else line_cmap,
                alpha=line_alpha, lw=line_width,
                **lineplot_kwargs
            )
            if line_hue is None:
                for line in plt.gca().get_lines():
                    line.set_color(line_color)
            # plt.show()
            
        elif line_hue == 'decline_agreement':
            # code from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
            for lenient_id in lenient_ids:
                plot_data = self.contractions[self.contractions['human_id'] == lenient_id]
                i_x = plot_data['decline_rate']
                i_y = plot_data[f'contraction_{y}']
                i_hue = plot_data['decline_agreement']
                points = np.array([i_x, i_y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                norm = plt.Normalize(0, 1)
                lc = LineCollection(
                    segments,
                    zorder=-1,
                    cmap='Blues' if line_cmap is None else line_cmap,
                    norm=norm, linewidths=line_width
                )
                lc.set_array(i_hue)  # sets colors
                line = plt.gca().add_collection(lc)

            cbar = plt.gcf().colorbar(line)
            cbar.set_label('Decline agreement', rotation=270, labelpad=15)

        sns.scatterplot(
            data=self.human_stats[self.human_stats['human_id'].isin(evaluatee_ids)],
            x='decline_rate', y=y,
            zorder=1,
            color=scatter_color, marker=scatter_marker, s=scatter_size,
            **scatterplot_kwargs
        )
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.gca().set_xticklabels([f'{x:.0%}' for x in plt.gca().get_xticks()])
        if y == 'false_neg':
            plt.gca().set_yticklabels([f'{x:.1%}' for x in plt.gca().get_yticks()])
        plt.xlabel('Decline rate')
        plt.ylabel({'false_neg': 'False negatives (%)', 'fairness_ratio_log2': 'Fairness ratio (log2)'}[y])
        plt.title(title)
        if line_hue is None:
            plt.gca().get_legend().remove()
        elif line_hue == 'volume':
            plt.legend(title='Volume')

        if show:
            plt.show()
        else:
            return ax

    def plot_fairness_tradeoff(self, n_bins=20):
        binning_params = {
            'bins': np.linspace(0, 1, num=n_bins),
            'labels': np.linspace(0, 1, num=n_bins - 1)
        }

        contraction_to_avg_cols = (
            ['contraction_false_neg'] +
            (['contraction_fairness_ratio_log2'] if self.FAIRNESS_AWARE else [])
        )
        contraction_binned_stats = self.average_by_bins(
            data=self.contractions,
            bin_col='decline_rate',
            to_average_cols=contraction_to_avg_cols,
            **binning_params
        )

        humans_to_avg_cols = (
            ['false_neg'] +
            (['fairness_ratio_log2'] if self.FAIRNESS_AWARE else [])
        )
        human_binned_stats = self.average_by_bins(
            data=self.human_stats,
            bin_col='decline_rate',
            to_average_cols=humans_to_avg_cols,
            **binning_params
        )

        binned_stats = contraction_binned_stats.merge(
            human_binned_stats.add_prefix('human_'),
            left_index=True, right_index=True,
        )
        binned_stats['false_neg_dif'] = binned_stats['contraction_false_neg'] - binned_stats['human_false_neg']
        binned_stats['fairness_ratio_log2_dif'] = \
            binned_stats['contraction_fairness_ratio_log2'] - binned_stats['human_fairness_ratio_log2']

        sns.scatterplot(data=binned_stats, x='false_neg_dif', y='fairness_ratio_log2_dif')
        plt.show()
        
        plt.plot(binned_stats['contraction_false_neg'], binned_stats['contraction_fairness_ratio_log2'], label='contraction')
        plt.plot(binned_stats['human_false_neg'], binned_stats['human_fairness_ratio_log2'], label='human')
        plt.gca().set_xticklabels([f'{x:.1%}' for x in plt.gca().get_xticks()])
        plt.xlabel('False negatives (%)')
        plt.ylabel('Fairness ratio')
        plt.title('Fairness-performance trade-off through decline rates')
        plt.legend()
        plt.show()

        return binned_stats

    # setters ----------------------------------------------------------------------------------------------------------
    def set_contractions(self, contractions=None):
        if contractions is None:
            contractions = self.calc_contractions()
        self.contractions = contractions

    # getters ----------------------------------------------------------------------------------------------------------
    def get_humans(self, min_volume: int = 0, leniency_rank: int = None) -> list:
        filtered_human_stats = (
            self.human_stats[self.human_stats['volume'] >= min_volume]
            .sort_values(by='decline_rate', ascending=True)
            .reset_index(drop=True)
        )
        if leniency_rank is None:
            humans = filtered_human_stats['human_id'].tolist()
        else:
            humans = filtered_human_stats.loc[[leniency_rank-1], 'human_id'].tolist()

        return humans

    # static methods ---------------------------------------------------------------------------------------------------
    @staticmethod
    def average_by_bins(data: pd.DataFrame, bin_col: str, to_average_cols: list, bins=None, labels=None):
        return data.groupby(pd.cut(data[bin_col], bins=bins, labels=labels))[to_average_cols].mean()
