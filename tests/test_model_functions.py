import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from woe_scoring.core.model.functions import calc_model_results, save_scorecard
from woe_scoring.core.model.model import Model # For type hinting and mocking structure

# --- Tests for calc_model_results ---

def test_calc_model_results_success():
    mock_model = MagicMock(spec=Model)
    mock_model.feature_names_ = ['WOE_feat1', 'WOE_feat2']
    mock_model.coef_ = np.array([0.123, -0.456])
    mock_model.intercept_ = 0.05
    mock_model.pvalues_ = np.array([0.001, 0.002])
    # Mocking as if intercept_pvalue_ is an attribute, adjust if it's a property
    type(mock_model).intercept_pvalue_ = MagicMock(return_value=0.05) # For getattr default

    results_df = calc_model_results(mock_model)

    expected_df = pd.DataFrame({
        'Feature': ['const', 'WOE_feat1', 'WOE_feat2'],
        'coef': [0.05, 0.123, -0.456],
        'P>|z|': [0.05, 0.001, 0.002] # Using the mocked intercept_pvalue
    })
    pd.testing.assert_frame_equal(results_df, expected_df)

def test_calc_model_results_success_no_intercept_pvalue():
    mock_model = MagicMock(spec=Model)
    mock_model.feature_names_ = ['WOE_feat1']
    mock_model.coef_ = np.array([0.789])
    mock_model.intercept_ = -0.1
    mock_model.pvalues_ = np.array([0.003])
    # Ensure intercept_pvalue_ is not present, getattr will use default 0.0
    delattr(mock_model, 'intercept_pvalue_') # Ensure it's not on the mock
    if hasattr(mock_model, 'intercept_pvalue_'): # defensive, if delattr fails on mock
        del mock_model.intercept_pvalue_

    results_df = calc_model_results(mock_model)

    expected_df = pd.DataFrame({
        'Feature': ['const', 'WOE_feat1'],
        'coef': [-0.1, 0.789],
        'P>|z|': [0.0, 0.003] # Default 0.0 for const p-value
    })
    pd.testing.assert_frame_equal(results_df, expected_df)


def test_calc_model_results_missing_attributes():
    mock_model_incomplete = MagicMock(spec=Model)
    # Missing feature_names_
    mock_model_incomplete.coef_ = np.array([0.1])
    mock_model_incomplete.intercept_ = 0.01
    mock_model_incomplete.pvalues_ = np.array([0.1])

    with pytest.raises(AttributeError, match="Model object is missing required attributes"):
        calc_model_results(mock_model_incomplete)

def test_calc_model_results_mismatched_lengths():
    mock_model_mismatch = MagicMock(spec=Model)
    mock_model_mismatch.feature_names_ = ['WOE_feat1', 'WOE_feat2'] # 2 features
    mock_model_mismatch.coef_ = np.array([0.1]) # 1 coefficient
    mock_model_mismatch.intercept_ = 0.01
    mock_model_mismatch.pvalues_ = np.array([0.1, 0.2]) # 2 p-values (ok with features)
    type(mock_model_mismatch).intercept_pvalue_ = MagicMock(return_value=0.05)

    with pytest.raises(ValueError, match="Model's feature_names_ and coef_ have inconsistent lengths."):
        calc_model_results(mock_model_mismatch)

    mock_model_mismatch.coef_ = np.array([0.1, 0.2]) # Correct coef length
    mock_model_mismatch.pvalues_ = np.array([0.1]) # Incorrect pvalues length
    with pytest.raises(ValueError, match="Model's feature_names_ and pvalues_ have inconsistent lengths."):
        calc_model_results(mock_model_mismatch)


# --- Tests for save_scorecard (the function in .model.functions) ---

@patch('woe_scoring.core.model.functions.scorecard_builder.build_scorecard_data')
@patch('woe_scoring.core.model.functions.excel_builder.build_excel_scorecard_sheet')
@patch('pandas.ExcelWriter') # Mock pd.ExcelWriter
@patch('os.makedirs') # Mock os.makedirs
def test_save_scorecard_success(mock_makedirs, mock_excel_writer,
                                mock_build_excel_sheet, mock_build_scorecard_data, tmp_path):
    woe_feat_names = ['WOE_feat1', 'WOE_feat2']
    woe_rules = [
        {'feat1': [{'bin': [0,1], 'woe': 0.1}], 'type_feature': 'num'},
        {'feat2': [{'bin': ['A'], 'woe': -0.2}], 'type_feature': 'cat'}
    ]
    model_coeffs = pd.DataFrame({
        'Feature': ['const', 'WOE_feat1', 'WOE_feat2'],
        'coef': [0.05, 0.5, -0.5],
        'P>|z|': [0.1, 0.01, 0.02]
    })
    base_pts, odds_val, pdo_val = 600, 50, 20
    output_dir = tmp_path / "scorecards"

    mock_feature_dfs = [pd.DataFrame({'Score_feat1': [10]}), pd.DataFrame({'Score_feat2': [20]})]
    mock_build_scorecard_data.return_value = mock_feature_dfs

    # Mock the ExcelWriter context manager
    mock_writer_instance = MagicMock()
    mock_excel_writer.return_value.__enter__.return_value = mock_writer_instance

    save_scorecard(
        woe_feature_names_in_model=woe_feat_names,
        woe_encoder_rules=woe_rules,
        model_coeff_pvalue_df=model_coeffs,
        base_points=base_pts, odds=odds_val, points_to_double_odds=pdo_val,
        output_path=str(output_dir)
    )

    expected_factor = pdo_val / np.log(2)
    expected_offset = base_pts - expected_factor * np.log(odds_val)

    mock_build_scorecard_data.assert_called_once_with(
        woe_feature_names_in_model=woe_feat_names,
        woe_encoder_rules=woe_rules,
        model_results_df=model_coeffs,
        factor=pytest.approx(expected_factor),
        offset=pytest.approx(expected_offset)
    )

    mock_makedirs.assert_called_with(output_dir, exist_ok=True)
    expected_excel_path = output_dir / "Scorecard.xlsx"
    mock_excel_writer.assert_called_once_with(str(expected_excel_path), engine="xlsxwriter")

    mock_build_excel_sheet.assert_called_once_with(
        all_feature_scorecard_data=mock_feature_dfs,
        excel_writer=mock_writer_instance
    )

@patch('woe_scoring.core.model.functions.logger.error') # Check that error is logged
def test_save_scorecard_exception_handling(mock_logger_error, tmp_path):
    with patch('woe_scoring.core.model.functions.scorecard_builder.build_scorecard_data',
               side_effect=Exception("BuildDataError")):
        save_scorecard(
            woe_feature_names_in_model=[], woe_encoder_rules=[], model_coeff_pvalue_df=pd.DataFrame(),
            base_points=600, odds=50, points_to_double_odds=20, output_path=str(tmp_path)
        )
        mock_logger_error.assert_called_once()
        assert "BuildDataError" in mock_logger_error.call_args[0][0]

    with patch('woe_scoring.core.model.functions.excel_builder.build_excel_scorecard_sheet',
               side_effect=Exception("ExcelBuildError")):
        save_scorecard(
            woe_feature_names_in_model=[], woe_encoder_rules=[], model_coeff_pvalue_df=pd.DataFrame(),
            base_points=600, odds=50, points_to_double_odds=20, output_path=str(tmp_path)
        )
        # build_scorecard_data would have been called again, so logger_error will be called more than once total
        # We check the content of the latest call
        assert "ExcelBuildError" in mock_logger_error.call_args[0][0]
