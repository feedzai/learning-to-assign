import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def inplace_one_hot_encode(df, cols_to_ohe: list, drop=True):
    """
    One-hot encode "cols_to_ohe" and return dataframe.
    """
    ohe = OneHotEncoder()
    ohe_ft_arr = ohe.fit_transform(df.loc[:, cols_to_ohe]).toarray()
    ohe_ft_df = pd.DataFrame(
        ohe_ft_arr,
        index=df.index,
        columns=ohe.get_feature_names()
    )

    df = pd.concat(objs=(df, ohe_ft_df), axis=1)
    if drop:
        df = df.drop(columns=cols_to_ohe)

    return df


class InplaceOneHotEncoder:

    def __init__(self, categorical_cols, **kwargs):
        self.ohe = OneHotEncoder(**kwargs)
        self.categorical_cols = categorical_cols

    def fit(self, X, *args, **kwargs):
        self.ohe.fit(X.loc[:, self.categorical_cols])

    def transform(self, X, *args, **kwargs):
        ohe_ft_arr = self.ohe.transform(X.loc[:, self.categorical_cols]).toarray()
        ohe_ft_df = pd.DataFrame(
            ohe_ft_arr,
            index=X.index,
            columns=self.ohe.get_feature_names(self.categorical_cols)  # update to .get_feature_names_out() after v1.0
        )
        X_with_ohe = pd.concat(objs=(X, ohe_ft_df), axis=1).drop(columns=self.categorical_cols)

        return X_with_ohe

    def fit_transform(self, X, *args, **kwargs):
        self.fit(X)
        return self.transform(X)

class MachineLearningPreprocessor:

    def __init__(self, data_types, standardize_numericals, one_hot_encode_categoricals):
        self.dtypes = data_types
        self.standardize_numericals = standardize_numericals
        self.one_hot_encode_categoricals = one_hot_encode_categoricals

        self.std_scaler = StandardScaler()
        self.one_hot_enc = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse=False)
        self.mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.fill_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='null')

    def process_numericals(self, x: pd.DataFrame, is_train: bool):

        # treat infinities as null values
        x = x.replace([np.inf, -np.inf], np.nan)

        # impute missing values with the mean
        if is_train:
            x = self.mean_imputer.fit_transform(x)
        else:
            x = self.mean_imputer.transform(x)
        x = pd.DataFrame(x, columns=self.mean_imputer.feature_names_in_)

        if self.standardize_numericals:
            # standardize values
            if is_train:
                x = self.std_scaler.fit_transform(x)
            else:
                x = self.std_scaler.transform(x)
            x = pd.DataFrame(x, columns=self.std_scaler.get_feature_names_out())

            # add marker for standardized values
            x.columns = [f'{c}_scaled' for c in x.columns]

        x = x.reset_index(drop=True)  # already affected by previous changes

        return x

    def process_categoricals(self, x: pd.DataFrame, is_train: bool):
        if not self.one_hot_encode_categoricals:
            raise NotImplementedError()
        # treat infinities as null values
        x = x.replace([np.inf, -np.inf], np.nan)

        # impute missing values with a new category
        if is_train:
            x = self.fill_imputer.fit_transform(x)
        else:
            x = self.fill_imputer.transform(x)
        x = pd.DataFrame(x, columns=self.fill_imputer.feature_names_in_)

        # standardize values
        if is_train:
            x = self.one_hot_enc.fit_transform(x)
        else:
            x = self.one_hot_enc.transform(x)
        x = pd.DataFrame(x, columns=self.one_hot_enc.get_feature_names_out())
        x = x.reset_index(drop=True)  # already affected by previous changes

        return x

    def fit_transform(self, train):

        proc_train = pd.concat(
            objs=(
                train[self.dtypes['additional_fields']].reset_index(drop=True),
                self.process_numericals(train[self.dtypes['numerical_fields']], is_train=True),
                self.process_categoricals(train[self.dtypes['categorical_fields']], is_train=True)
            ),
            axis=1
        )
        # when numericals are not standardized, there might be duplicates in the additional fields
        proc_train = proc_train.loc[:, ~proc_train.columns.duplicated()]

        proc_train.index = train.index
        
        return proc_train

    def transform(self, test):

        proc_test = pd.concat(
            objs=(
                test[self.dtypes['additional_fields']].reset_index(drop=True),
                self.process_numericals(test[self.dtypes['numerical_fields']], is_train=False),
                self.process_categoricals(test[self.dtypes['categorical_fields']], is_train=False)
            ),
            axis=1
        )
        # when numericals are not standardized, there might be duplicates in the additional fields
        proc_test = proc_test.loc[:, ~proc_test.columns.duplicated()]

        proc_test.index = test.index

        return proc_test
