import pytest
import pandas as pd
import numpy as np
from woe_scoring import WOETransformer


def test_woe_transformer_initialization():

    transformer = WOETransformer()
    assert transformer.max_bins == 10
    assert transformer.min_pct_group == 0.05
    assert transformer.n_jobs == 1
    assert transformer.prefix == "WOE_"

def test_woe_transformer_fit(sample_data):
    df, y = sample_data
    transformer = WOETransformer(cat_features=['categorical_feature'])

    transformer.fit(df, y)

    assert len(transformer.woe_iv_dict) == 2  # One for each feature
    assert transformer.feature_names == ['numeric_feature', 'categorical_feature']

def test_woe_transformer_transform(sample_data, woe_transformer):
    df, y = sample_data

    woe_transformer.fit(df, y)
    transformed_df = woe_transformer.transform(df)

    assert isinstance(transformed_df, pd.DataFrame)
    assert all(col.startswith('WOE_') for col in transformed_df.columns)
    assert transformed_df.shape[1] == 2  # Should have same number of features
    assert not transformed_df.isna().any().any()  # Should not have any NaN values

def test_woe_transformer_with_missing_values(sample_data):
    df, y = sample_data

    # Add some missing values
    df.loc[0:10, 'numeric_feature'] = np.nan
    df.loc[20:30, 'categorical_feature'] = np.nan

    transformer = WOETransformer(cat_features=['categorical_feature'])
    transformer.fit(df, y)
    transformed_df = transformer.transform(df)

    assert not transformed_df.isna().any().any()  # Should handle missing values
