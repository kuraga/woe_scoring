import pytest
import pandas as pd
import numpy as np

def test_create_model_initialization():
    from woe_scoring import CreateModel

    model = CreateModel()
    assert model.model_params['selection_method'] == 'rfe'
    assert model.model_params['model_type'] == 'sklearn'
    assert model.model_params['class_weight'] == 'balanced'

def test_create_model_fit_predict(sample_data, woe_transformer):
    from woe_scoring import CreateModel

    df, y = sample_data

    # First transform data using WOE
    woe_transformer.fit(df, y)
    woe_df = woe_transformer.transform(df)

    # Fit model
    model = CreateModel()
    model.fit(woe_df, y)

    # Test predictions
    predictions = model.predict(woe_df)
    probabilities = model.predict_proba(woe_df)

    assert len(predictions) == len(y)
    assert all(isinstance(p, (np.int64, np.int32, int)) for p in predictions)
    assert probabilities.shape == (len(y), 2)
    assert all(0 <= p <= 1 for p in probabilities.ravel())

def test_create_model_feature_selection(sample_data, woe_transformer):
    from woe_scoring import CreateModel

    df, y = sample_data

    # Transform data
    woe_transformer.fit(df, y)
    woe_df = woe_transformer.transform(df)

    # Test with different selection methods
    for method in ['rfe', 'sfs', 'iv']:
        model = CreateModel(selection_method=method)
        model.fit(woe_df, y)

        assert model.model is not None
        assert hasattr(model.model, 'feature_selector')
