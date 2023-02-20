import matplotlib.pyplot as plt
import numpy as np
from shap import summary_plot as shap_summary_plot


def analyst_shap_summary(analyst_id, shap_values, raw_data, model_data, features=None):
    all_features_arr = np.array(model_data.columns.tolist())
    if features is not None:
        features_ix = np.isin(all_features_arr, test_elements=features)
    else:
        features_ix = np.full(shape=all_features_arr.shape, fill_value=True, dtype=bool)

    selected_analyst_ix = (raw_data['analyst_id'] == analyst_id).values
    shap_summary_plot(
        shap_values[selected_analyst_ix][:, features_ix],
        model_data.loc[selected_analyst_ix, features_ix].values,
        feature_names=model_data.loc[selected_analyst_ix, features_ix].columns.tolist(),
        plot_type='violin',
        show=False
    )
    plt.gcf().axes[-1].set_aspect(50)
    plt.gcf().axes[-1].set_box_aspect(50)
    plt.show()  # no option to return since the figure must be fixed as per the two above lines
