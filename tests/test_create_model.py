import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from woe_scoring.core.main import CreateModel # Direct import for clarity in tests
from woe_scoring.core.model.model import Model # Needed for type checking if not mocking entirely
from woe_scoring.core.model.selector import FeatureSelector # Needed for type checking

# Sample data fixtures (can be moved to conftest.py if shared)
@pytest.fixture
def sample_woe_data():
    return pd.DataFrame({
        'WOE_feat1': np.random.rand(100) - 0.5,
        'WOE_feat2': np.random.rand(100) - 0.5,
        'WOE_feat3': np.random.rand(100) - 0.5
    })

@pytest.fixture
def sample_target():
    return pd.Series(np.random.randint(0, 2, 100))

@pytest.fixture
def sample_woe_rules():
    return [
        {"feat1": [{"bin": [0,1], "woe": 0.1, "iv": 0.01, "bad":10, "total":50, "pct": 0.5, "bad_rate":0.2}], "type_feature": "num", "missing_bin": "first"},
        {"feat2": [{"bin": ['A','B'], "woe": -0.2, "iv": 0.02, "bad":5, "total":20, "pct": 0.2, "bad_rate":0.25}], "type_feature": "cat", "missing_bin": "last"},
        {"feat3": [{"bin": [2,3], "woe": 0.3, "iv": 0.03, "bad":15, "total":30, "pct": 0.3, "bad_rate":0.5}], "type_feature": "num", "missing_bin": "first"},
    ]

# --- Initialization Tests ---
def test_createmodel_initialization_defaults():
    cm = CreateModel()
    assert cm.selection_method == 'rfe'
    assert cm.model_type == 'sklearn'
    assert cm.max_vars is None
    assert cm.special_cols == []
    assert cm.unused_cols == []
    assert cm.n_jobs == 1
    assert cm.gini_threshold == 5.0
    assert cm.iv_threshold == 0.05
    assert cm.corr_threshold == 0.5
    assert cm.min_pct_group == 0.05
    assert cm.random_state is None
    assert cm.class_weight == 'balanced'
    assert cm.direction == "forward"
    assert cm.cv == 3
    assert cm.l1_exp_scale == 4
    assert cm.l1_grid_size == 20
    assert cm.scoring == "roc_auc"
    assert cm.woe_transformer_rules == []
    assert cm.input_feature_names == []
    assert cm.logger is not None

def test_createmodel_initialization_custom_params(sample_woe_rules):
    cm = CreateModel(
        selection_method='iv', model_type='statsmodels', max_vars=10,
        special_cols=['id'], unused_cols=['temp'], n_jobs=2,
        gini_threshold=10.0, iv_threshold=0.1, corr_threshold=0.6,
        min_pct_group=0.1, random_state=42, class_weight=None,
        direction='backward', cv=5, l1_exp_scale=3, l1_grid_size=10,
        scoring='accuracy', woe_transformer_rules=sample_woe_rules,
        feature_names=['feat1', 'feat2', 'feat3']
    )
    assert cm.selection_method == 'iv'
    assert cm.model_type == 'statsmodels'
    assert cm.max_vars == 10
    assert cm.special_cols == ['id']
    assert cm.unused_cols == ['temp']
    assert cm.woe_transformer_rules == sample_woe_rules
    assert cm.input_feature_names == ['feat1', 'feat2', 'feat3']
    assert cm.random_state == 42


# --- Fit Method Tests ---
@patch('woe_scoring.core.main.FeatureSelector')
@patch('woe_scoring.core.main.Model')
@patch('woe_scoring.core.main._calc_model_results')
def test_createmodel_fit_successful(mock_calc_results, mock_model_cls, mock_selector_cls,
                                   sample_woe_data, sample_target, sample_woe_rules):
    # Setup mocks
    mock_selector_instance = MagicMock(spec=FeatureSelector)
    mock_selector_instance.select.return_value = ['WOE_feat1', 'WOE_feat2']
    mock_selector_cls.return_value = mock_selector_instance

    mock_model_instance = MagicMock(spec=Model)
    # Mock attributes that would be set after fitting
    mock_model_instance.feature_names_ = ['WOE_feat1', 'WOE_feat2'] # Names of features used in model
    mock_model_instance.coef_ = np.array([0.5, -0.5])
    mock_model_instance.intercept_ = 0.1
    mock_model_instance.pvalues_ = np.array([0.01, 0.02])
    # mock_model_instance.intercept_pvalue_ = 0.05 # If Model class provides this
    type(mock_model_instance).intercept_pvalue_ = PropertyMock(return_value=0.05)


    mock_model_cls.return_value = mock_model_instance

    mock_calc_results.return_value = pd.DataFrame({'Feature': ['const', 'WOE_feat1', 'WOE_feat2'],
                                                   'coef': [0.1, 0.5, -0.5],
                                                   'P>|z|': [0.05, 0.01, 0.02]})

    cm = CreateModel(woe_transformer_rules=sample_woe_rules, feature_names=['feat1', 'feat2', 'feat3'])
    cm.fit(sample_woe_data, sample_target)

    # Assert FeatureSelector was called correctly
    mock_selector_cls.assert_called_once()
    selector_args = mock_selector_cls.call_args[1] # Get kwargs
    assert selector_args['feature_names'] == ['feat1', 'feat2', 'feat3'] # Original names
    assert selector_args['woe_rules'] == sample_woe_rules
    mock_selector_instance.select.assert_called_once_with(sample_woe_data, sample_target, list(sample_woe_data.columns))

    # Assert Model was called correctly
    mock_model_cls.assert_called_once()
    mock_model_instance.get_model.assert_called_once()
    # Check that data passed to get_model has only selected features
    df_passed_to_model = mock_model_instance.get_model.call_args[0][0]
    pd.testing.assert_frame_equal(df_passed_to_model, sample_woe_data[['WOE_feat1', 'WOE_feat2']])

    assert cm.feature_selector == mock_selector_instance
    assert cm.model == mock_model_instance

    # Assert _calc_model_results was called with the fitted model instance
    mock_calc_results.assert_called_once_with(mock_model_instance)
    assert cm.model_results is not None
    assert not cm.model_results.empty


@patch('woe_scoring.core.main.FeatureSelector')
def test_createmodel_fit_no_features_selected(mock_selector_cls, sample_woe_data, sample_target):
    mock_selector_instance = MagicMock()
    mock_selector_instance.select.return_value = [] # No features selected
    mock_selector_cls.return_value = mock_selector_instance

    cm = CreateModel()
    with pytest.raises(ValueError, match="Feature selection returned no features"):
        cm.fit(sample_woe_data, sample_target)


@patch('woe_scoring.core.main.FeatureSelector', side_effect=Exception("Selector Error"))
def test_createmodel_fit_selector_exception(mock_selector_cls_exc, sample_woe_data, sample_target):
    cm = CreateModel()
    with pytest.raises(RuntimeError, match="Feature selection failed: Selector Error"):
        cm.fit(sample_woe_data, sample_target)

@patch('woe_scoring.core.main.FeatureSelector')
@patch('woe_scoring.core.main.Model', side_effect=Exception("Model Error"))
def test_createmodel_fit_model_exception(mock_model_cls_exc, mock_selector_cls, sample_woe_data, sample_target):
    mock_selector_instance = MagicMock()
    mock_selector_instance.select.return_value = ['WOE_feat1'] # Selector works
    mock_selector_cls.return_value = mock_selector_instance

    cm = CreateModel()
    with pytest.raises(RuntimeError, match="Model fitting failed: Model Error"):
        cm.fit(sample_woe_data, sample_target)

# --- Predict/PredictProba Method Tests ---
@patch('woe_scoring.core.main.FeatureSelector')
@patch('woe_scoring.core.main.Model')
def test_createmodel_predict_proba_predict(mock_model_cls, mock_selector_cls, sample_woe_data, sample_target):
    # Fit the model (mocked)
    mock_selector_instance = MagicMock()
    mock_selector_instance.select.return_value = ['WOE_feat1']
    mock_selector_cls.return_value = mock_selector_instance

    mock_model_instance = MagicMock()
    mock_model_instance.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
    mock_model_instance.predict.return_value = np.array([0, 1])
    mock_model_cls.return_value = mock_model_instance

    cm = CreateModel()
    cm.fit(sample_woe_data, sample_target) # Fit with mocks

    # Test predict_proba
    test_data = pd.DataFrame({'WOE_feat1': [0.1, 0.2]}) # Dummy data for prediction
    proba_results = cm.predict_proba(test_data)
    mock_model_instance.predict_proba.assert_called_once_with(test_data)
    np.testing.assert_array_equal(proba_results, np.array([[0.8, 0.2], [0.1, 0.9]]))

    # Test predict
    predict_results = cm.predict(test_data)
    mock_model_instance.predict.assert_called_once_with(test_data)
    np.testing.assert_array_equal(predict_results, np.array([0, 1]))

def test_createmodel_predict_before_fit():
    cm = CreateModel()
    with pytest.raises(ValueError, match="Model must be fitted before prediction."):
        cm.predict_proba(pd.DataFrame({'A':[1]}))
    with pytest.raises(ValueError, match="Model must be fitted before prediction."):
        cm.predict(pd.DataFrame({'A':[1]}))


# --- Placeholder for save_reports, generate_sql, save_scorecard tests ---
# These will require more specific mocking of the model object and external functions.
# For example, for save_reports, if model_type is 'statsmodels',
# self.model.model.model_ (the statsmodels result wrapper) needs to be mocked.
# For generate_sql and save_scorecard, the 'encoder' argument (WOETransformer.woe_iv_dict)
# and other model attributes (coef_, intercept_, feature_names_) are important.
# And the respective functions from .model.model_analyzer, .model.sql_generator,
# and .model.functions need to be patched.

@patch('woe_scoring.core.main.model_analyzer.save_model_reports')
@patch('woe_scoring.core.main.FeatureSelector')
@patch('woe_scoring.core.main.Model')
def test_createmodel_save_reports_statsmodels(mock_model_cls, mock_selector_cls, mock_save_reports_func,
                                            sample_woe_data, sample_target, tmp_path):
    # Setup for a statsmodels type model
    mock_selector_instance = MagicMock()
    mock_selector_instance.select.return_value = ['WOE_feat1']
    mock_selector_cls.return_value = mock_selector_instance

    mock_model_instance = MagicMock()
    mock_model_instance.model_type = 'statsmodels' # Critical for save_reports path
    # Mock the nested structure for statsmodels results
    mock_sm_results = MagicMock()
    mock_sm_results.summary.return_value.as_text.return_value = "Statsmodels Summary Text"
    mock_sm_results.wald_test_terms.return_value.summary_frame.return_value = pd.DataFrame({'test': [1]})

    # self.model.model is the wrapper (e.g. SMWrapper), self.model.model.model_ is the actual statsmodels result
    mock_model_wrapper = MagicMock()
    mock_model_wrapper.model_ = mock_sm_results # This is what actual_sm_model_obj points to
    mock_model_instance.model = mock_model_wrapper

    mock_model_cls.return_value = mock_model_instance

    cm = CreateModel(model_type='statsmodels')
    cm.fit(sample_woe_data, sample_target) # Fit with mocks

    report_path = str(tmp_path / "reports")
    cm.save_reports(report_path)

    mock_save_reports_func.assert_called_once_with(
        model_summary_text="Statsmodels Summary Text",
        wald_test_summary_df=pd.DataFrame({'test': [1]}),
        path=report_path
    )

@patch('builtins.print') # Assuming it prints for sklearn, or use logger if CreateModel uses it here
@patch('woe_scoring.core.main.FeatureSelector')
@patch('woe_scoring.core.main.Model')
def test_createmodel_save_reports_sklearn(mock_model_cls, mock_selector_cls, mock_print_func,
                                           sample_woe_data, sample_target):
    # Test that it skips for sklearn or logs appropriately
    mock_selector_instance = MagicMock()
    mock_selector_instance.select.return_value = ['WOE_feat1']
    mock_selector_cls.return_value = mock_selector_instance

    mock_model_instance = MagicMock()
    mock_model_instance.model_type = 'sklearn' # Ensure model_type is sklearn
    mock_model_instance.model = MagicMock() # Mock the inner model object
    mock_model_cls.return_value = mock_model_instance

    cm = CreateModel(model_type='sklearn')
    cm.fit(sample_woe_data, sample_target)

    cm.save_reports("some_path")
    mock_print_func.assert_called_with("Skipping saving statsmodels-specific reports for model type: sklearn")


def test_createmodel_save_reports_before_fit():
    cm = CreateModel()
    with pytest.raises(ValueError, match="Model must be fitted before saving reports"):
        cm.save_reports("dummy_path")


@patch('woe_scoring.core.main.sql_generator.generate_sql_query')
@patch('woe_scoring.core.main.FeatureSelector')
@patch('woe_scoring.core.main.Model')
def test_createmodel_generate_sql(mock_model_cls, mock_selector_cls, mock_generate_sql,
                                 sample_woe_data, sample_target, sample_woe_rules):
    mock_selector_instance = MagicMock()
    mock_selector_instance.select.return_value = ['WOE_feat1', 'WOE_feat2']
    mock_selector_cls.return_value = mock_selector_instance

    mock_model_instance = MagicMock()
    mock_model_instance.feature_names_ = ['WOE_feat1', 'WOE_feat2']
    mock_model_instance.coef_ = np.array([0.5, -0.5])
    mock_model_instance.intercept_ = 0.1
    mock_model_cls.return_value = mock_model_instance

    mock_generate_sql.return_value = "SELECT SQL QUERY;"

    cm = CreateModel(woe_transformer_rules=sample_woe_rules)
    cm.fit(sample_woe_data, sample_target)

    sql_query = cm.generate_sql(encoder=cm.woe_transformer_rules)

    mock_generate_sql.assert_called_once_with(
        woe_encoder_info=sample_woe_rules,
        woe_feature_names=['WOE_feat1', 'WOE_feat2'],
        model_coefficients=np.array([0.5, -0.5]),
        model_intercept=0.1
    )
    assert sql_query == "SELECT SQL QUERY;"


def test_createmodel_generate_sql_before_fit():
    cm = CreateModel()
    with pytest.raises(ValueError, match="Model must be fitted before generating SQL"):
        cm.generate_sql(encoder=[])


@patch('woe_scoring.core.main._save_scorecard')
@patch('woe_scoring.core.main.FeatureSelector')
@patch('woe_scoring.core.main.Model')
@patch('woe_scoring.core.main._calc_model_results')
def test_createmodel_save_scorecard(mock_calc_results, mock_model_cls, mock_selector_cls, mock_save_scorecard_func,
                                  sample_woe_data, sample_target, sample_woe_rules, tmp_path):
    mock_selector_instance = MagicMock()
    mock_selector_instance.select.return_value = ['WOE_feat1', 'WOE_feat2']
    mock_selector_cls.return_value = mock_selector_instance

    mock_model_instance = MagicMock()
    mock_model_instance.feature_names_ = ['WOE_feat1', 'WOE_feat2']
    mock_model_instance.coef_ = np.array([0.5, -0.5])
    mock_model_instance.intercept_ = 0.1
    mock_model_instance.pvalues_ = np.array([0.01, 0.02])
    type(mock_model_instance).intercept_pvalue_ = PropertyMock(return_value=0.05) # For _calc_model_results
    mock_model_cls.return_value = mock_model_instance

    mock_model_results_df = pd.DataFrame({'Feature': ['const', 'WOE_feat1', 'WOE_feat2'],
                                          'coef': [0.1, 0.5, -0.5],
                                          'P>|z|': [0.05, 0.01, 0.02]})
    mock_calc_results.return_value = mock_model_results_df

    cm = CreateModel(woe_transformer_rules=sample_woe_rules)
    cm.fit(sample_woe_data, sample_target)

    assert cm.model_results is mock_model_results_df

    scorecard_path = str(tmp_path)
    cm.save_scorecard(encoder=cm.woe_transformer_rules, path=scorecard_path,
                      base_scorecard_points=600, odds=50, points_to_double_odds=20)

    mock_save_scorecard_func.assert_called_once_with(
        woe_feature_names_in_model=['WOE_feat1', 'WOE_feat2'],
        woe_encoder_rules=sample_woe_rules,
        model_coeff_pvalue_df=mock_model_results_df,
        base_points=600,
        odds=50,
        points_to_double_odds=20,
        output_path=scorecard_path
    )

def test_createmodel_save_scorecard_before_fit():
    cm = CreateModel()
    with pytest.raises(ValueError, match="Model must be fitted and results calculated before saving scorecard"):
        cm.save_scorecard(encoder=[], path=".")
