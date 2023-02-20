import pandas as pd


def undersample_value_for_attribute(
        df: pd.DataFrame,
        attribute_col: str, value_col: str, value_target: float,
        random_state: int = 42):
    """
    Returns undersampled dataframe that ensures that for each unique attribute,
    the prevalence of unique values of another column follows specified fractions.
    :param df: the pandas dataframe to be transformed.
    :param attribute_col: string with name of the column containing the attribute.
    :param value_col: string with name of the column containing the values.
    :param value_target: target mean value of the value_col.
    :param random_state: random seed for pandas.sample.
    :return: transformed pandas dataframe.
    """
    sampled_dfs_list = list()
    unique_attrs = df[attribute_col].unique()

    for attr in unique_attrs:
        attr_filter = (df[attribute_col] == attr)
        attr_df = df[attr_filter]
        observed_mean = attr_df[value_col].mean()

        # too extreme cannot be worked with
        if (observed_mean < 0.05) or (observed_mean > 0.95):
            continue

        if observed_mean > value_target:
            negatives_n = (attr_df[value_col] == 0).sum()
            new_positives_n = int(
                negatives_n /
                ((1/value_target)-1)
            )

            sampled_df_attr_positives = attr_df[attr_df[value_col] == 1].sample(
                n=new_positives_n,
                replace=False,
                random_state=random_state
            )
            sampled_df_attr_negatives = attr_df[attr_df[value_col] == 0]

        else:
            positives_n = (attr_df[value_col] == 1).sum()
            new_negatives_n = int((positives_n / value_target) - positives_n)

            sampled_df_attr_negatives = attr_df[attr_df[value_col] == 0].sample(
                n=new_negatives_n,
                replace=False,
                random_state=random_state
            )
            sampled_df_attr_positives = attr_df[attr_df[value_col] == 1]

        sampled_df_attr = pd.concat(
            (sampled_df_attr_negatives, sampled_df_attr_positives),
            axis=0
        )
        sampled_dfs_list.append(sampled_df_attr)

    undersampled_df = pd.concat(sampled_dfs_list, axis=0)
    undersampled_df = undersampled_df.sample(frac=1).reset_index(drop=True)  # shuffle

    return undersampled_df

def shuffle_while_respecting(
        df: pd.DataFrame,
        cols_to_shuffle: list, cols_to_respect: list,
        random_state: int = 42):
    """
    Shuffle some columns, while not changing their relationship with some other columns (to respect).
    :param df: the pandas dataframe to be transformed.
    :param cols_to_shuffle: list of strings with the names of columns to be shuffled.
    :param cols_to_respect: list of strings with the names of columns to be respected.
    :param random_state: random seed for pandas.sample.
    :return: transformed pandas dataframe
    """
    uniques_combos = (
        df[cols_to_respect]
        .value_counts()
        .reset_index(name='count')
        .drop(columns='count')
    )
    shuffled_df = df.copy()

    for _, row in uniques_combos.iterrows():
        filtered_ix = (shuffled_df[cols_to_respect] == row).all(axis=1)
        shuffled_df.loc[filtered_ix, cols_to_shuffle] = (
            shuffled_df.loc[filtered_ix, cols_to_shuffle]
            .sample(frac=1, random_state=random_state)
            .values
        )
        random_state += 1  # in case there is some hidden unforeseeable structure

    return shuffled_df
