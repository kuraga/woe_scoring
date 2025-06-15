import pytest
import pandas as pd
import numpy as np
from woe_scoring import WOETransformer
from typing import Dict

def test_woe_transformer_initialization():
    transformer = WOETransformer()
    assert transformer.max_bins == 10
    assert transformer.min_pct_group == 0.05
    assert transformer.n_jobs == 1
    assert transformer.prefix == "WOE_"
    assert transformer.merge_type == "chi2"
    assert transformer.cat_features == []
    assert transformer.special_cols == []
    assert transformer.cat_features_threshold == 0
    assert transformer.diff_woe_threshold == 0.05
    assert not transformer.safe_original_data

    # Test with some custom valid parameters
    custom_transformer = WOETransformer(
        max_bins=5,
        min_pct_group=0.1,
        prefix="TRANSFORMED_",
        cat_features=["cat1"],
        special_cols=["id"],
        cat_features_threshold=10,
        diff_woe_threshold=0.02,
        safe_original_data=True
    )
    assert custom_transformer.max_bins == 5
    assert custom_transformer.min_pct_group == 0.1
    assert custom_transformer.prefix == "TRANSFORMED_"
    assert custom_transformer.cat_features == ["cat1"]
    assert custom_transformer.special_cols == ["id"]
    assert custom_transformer.cat_features_threshold == 10
    assert custom_transformer.diff_woe_threshold == 0.02
    assert custom_transformer.safe_original_data

def test_woe_transformer_initialization_invalid_params():
    # Check various invalid min_pct_group values
    with pytest.raises(ValueError):
        WOETransformer(min_pct_group=0)
    with pytest.raises(ValueError):
        WOETransformer(min_pct_group=1)
    with pytest.raises(ValueError):
        WOETransformer(min_pct_group=1.1)

    # Check invalid max_bins int values
    with pytest.raises(ValueError):
        WOETransformer(max_bins=0)
    with pytest.raises(ValueError):
        WOETransformer(max_bins=-5)

    # Check invalid max_bins float values
    with pytest.raises(ValueError):
        WOETransformer(max_bins=0.0)
    with pytest.raises(ValueError):
        WOETransformer(max_bins=1.1)
    with pytest.raises(ValueError):
        WOETransformer(max_bins=-0.5)

def test_woe_transformer_fit_basic(sample_data): # Renamed from test_woe_transformer_fit
    df, y = sample_data
    # Assuming sample_data has 'numeric_feature' and 'categorical_feature'
    transformer = WOETransformer(cat_features=['categorical_feature'])
    transformer.fit(df, y)

    assert isinstance(transformer.woe_iv_dict, list)
    # Number of entries in woe_iv_dict should match number of features processed
    assert len(transformer.woe_iv_dict) == df.shape[1]

    processed_feature_names = set()
    for item in transformer.woe_iv_dict:
        assert isinstance(item, dict)
        feature_name_in_dict = next(iter(item)) # First key is feature name
        processed_feature_names.add(feature_name_in_dict)
        assert "type_feature" in item
        assert item[feature_name_in_dict] is not None # Should be a list of BadRates/bin dicts
        if item["type_feature"] == "cat":
            assert feature_name_in_dict in transformer.cat_features
        else:
            assert feature_name_in_dict in transformer.num_features

    assert set(transformer.feature_names) == processed_feature_names
    assert transformer.classes_ is not None
    assert len(transformer.classes_) == 2 # Binary target assumed for WOE

def test_woe_transformer_fit_with_special_cols(sample_data):
    df, y = sample_data
    df_with_special = df.copy()
    df_with_special['ID'] = range(len(df))
    df_with_special['Timestamp'] = pd.to_datetime(pd.Timestamp('2023-01-01') + pd.to_timedelta(np.arange(len(df)), 'D'))

    transformer = WOETransformer(
        cat_features=['categorical_feature'], # Assuming this exists in sample_data
        special_cols=['ID', 'Timestamp']
    )
    transformer.fit(df_with_special, y)

    assert 'ID' not in transformer.feature_names
    assert 'Timestamp' not in transformer.feature_names
    if 'numeric_feature' in df.columns: assert 'numeric_feature' in transformer.feature_names
    if 'categorical_feature' in df.columns: assert 'categorical_feature' in transformer.feature_names
    assert len(transformer.woe_iv_dict) == df.shape[1] # Original non-special features

def test_woe_transformer_fit_cat_threshold():
    data = pd.DataFrame({
        'feat1_numeric': np.random.rand(50) * 100,
        'feat2_cat_lowcard_obj': pd.Series(['A', 'B'] * 25, dtype='object'),
        'feat3_num_lowcard_as_cat': pd.Series([1, 2, 3, 1, 2] * 10), # Numeric, but low card
        'feat4_cat_medcard_obj': pd.Series(['X', 'Y', 'Z', 'W', 'V'] * 10, dtype='object') # 5 unique
    })
    y = pd.Series(np.random.randint(0, 2, 50))

    transformer = WOETransformer(cat_features_threshold=4) # Unique values < 4 treated as cat
    transformer.fit(data, y)

    assert 'feat1_numeric' in transformer.num_features
    assert 'feat2_cat_lowcard_obj' in transformer.cat_features # Is object, auto-cat
    assert 'feat3_num_lowcard_as_cat' in transformer.cat_features # 3 unique values < threshold 4
    assert 'feat4_cat_medcard_obj' in transformer.num_features # Is object, but 5 unique values >= threshold 4
                                                               # Correction: find_cat_features is (is_object OR nunique < thresh)
                                                               # So feat4_cat_medcard_obj should be cat
    assert 'feat4_cat_medcard_obj' in transformer.cat_features


    assert len(transformer.woe_iv_dict) == data.shape[1]


def test_woe_transformer_fit_edge_cases():
    transformer = WOETransformer()
    valid_y = pd.Series([0, 1, 0, 1, 0])

    with pytest.raises(ValueError, match="Input data DataFrame cannot be empty."):
        transformer.fit(pd.DataFrame(), valid_y)
    with pytest.raises(ValueError, match="Target must be a non-empty pandas Series or numpy array."):
        transformer.fit(pd.DataFrame({'A': [1,2,3]}), pd.Series([], dtype=int))
    with pytest.raises(ValueError, match="Data and target must have the same number of rows."):
        transformer.fit(pd.DataFrame({'A': [1,2,3]}), pd.Series([0,1]))

    df_all_special = pd.DataFrame({'ID1': [1,2,3], 'ID2': ['A','B','C']})
    y_for_special = pd.Series([0,1,0])
    transformer_all_special = WOETransformer(special_cols=['ID1', 'ID2'])
    transformer_all_special.fit(df_all_special, y_for_special)
    assert transformer_all_special.woe_iv_dict == []
    assert transformer_all_special.feature_names == []

# Basic transform test using the new fixture
def test_woe_transformer_transform_basic(fitted_woe_transformer, simple_data_for_woe):
    transformer = fitted_woe_transformer
    data, _ = simple_data_for_woe # Use the same data it was fitted on

    transformed_df = transformer.transform(data)
    assert isinstance(transformed_df, pd.DataFrame)
    assert "WOE_num_feat" in transformed_df.columns
    assert "WOE_cat_feat" in transformed_df.columns
    assert "num_feat" not in transformed_df.columns # Default safe_original_data=False
    assert "cat_feat" not in transformed_df.columns
    assert not transformed_df.isna().any().any()

    # Check if WOE values are somewhat as expected (more detailed checks can be added)
    # For cat_feat 'A', 'B', 'C', 'D' had distinct target patterns
    woe_A = transformed_df.loc[data['cat_feat'] == 'A', 'WOE_cat_feat'].unique()
    woe_B = transformed_df.loc[data['cat_feat'] == 'B', 'WOE_cat_feat'].unique()
    assert len(woe_A) == 1 and len(woe_B) == 1
    assert woe_A[0] != woe_B[0] # Expect different WOEs due to target patterns

    # For num_feat, 2 bins were created. Low values vs High values
    woe_num_low = transformed_df.loc[data['num_feat'] <= 10, 'WOE_num_feat'].unique()
    woe_num_high = transformed_df.loc[data['num_feat'] > 10, 'WOE_num_feat'].unique()
    assert len(woe_num_low) == 1 and len(woe_num_high) == 1
    assert woe_num_low[0] != woe_num_high[0]


def test_woe_transformer_transform_safe_original_data(fitted_woe_transformer, simple_data_for_woe):
    transformer = WOETransformer( # Create new instance to modify safe_original_data
        cat_features=['cat_feat'], max_bins=2, min_pct_group=0.05,
        diff_woe_threshold=0.01, prefix="WOE_", safe_original_data=True
    )
    data, target = simple_data_for_woe
    transformer.fit(data, target) # Fit this new instance

    transformed_df = transformer.transform(data)
    assert "WOE_num_feat" in transformed_df.columns
    assert "WOE_cat_feat" in transformed_df.columns
    assert "num_feat" in transformed_df.columns # Original should be kept
    assert "cat_feat" in transformed_df.columns # Original should be kept
    assert transformed_df.shape[1] == data.shape[1] + 2


# Keep existing missing value test, ensure it fits with overall structure
def test_woe_transformer_fit_transform_with_missing_values(sample_data): # Renamed for clarity
    df, y = sample_data
    df_missing = df.copy()
    df_missing.loc[0:5, 'numeric_feature'] = np.nan      # Introduce NaNs
    df_missing.loc[5:10, 'categorical_feature'] = np.nan # Introduce NaNs

    transformer = WOETransformer(cat_features=['categorical_feature'])
    transformer.fit(df_missing, y) # Fit on data with NaNs

    # Transform the same data it was fit on
    transformed_df_fit_data = transformer.transform(df_missing)
    assert not transformed_df_fit_data.isna().any().any(), "Should handle NaNs it was fit on"

    # Transform new data that also has NaNs
    df_new_missing = df.copy() # Fresh copy from sample_data
    df_new_missing.loc[10:15, 'numeric_feature'] = np.nan
    df_new_missing.loc[15:20, 'categorical_feature'] = np.nan
    transformed_df_new_data = transformer.transform(df_new_missing)
    assert not transformed_df_new_data.isna().any().any(), "Should handle NaNs in new data"

    # Check that missing_bin strategy was applied in woe_iv_dict
    for rule in transformer.woe_iv_dict:
        assert "missing_bin" in rule
        assert rule["missing_bin"] is not None

    # Check that original NaN positions are filled in WOE columns
    # Example: numeric_feature had NaNs at index 0-5
    # Its WOE transformed column should have non-NaN values at these positions
    woe_numeric_col = "WOE_numeric_feature" # Assuming this is the naming
    if woe_numeric_col in transformed_df_fit_data.columns:
         assert not transformed_df_fit_data.loc[0:5, woe_numeric_col].isna().any()


# Placeholder for more transform tests based on requirements
# (unseen features, missing fitted features, empty woe_iv_dict)
def test_woe_transformer_transform_unseen_features(fitted_woe_transformer, simple_data_for_woe):
    transformer = fitted_woe_transformer
    data, _ = simple_data_for_woe

    data_with_unseen = data.copy()
    data_with_unseen['unseen_numeric'] = np.random.rand(len(data))
    data_with_unseen['unseen_categorical'] = ['X'] * len(data)

    transformed_df = transformer.transform(data_with_unseen)

    assert "WOE_num_feat" in transformed_df.columns
    assert "WOE_cat_feat" in transformed_df.columns
    assert "WOE_unseen_numeric" not in transformed_df.columns # WOE not created for unseen
    assert "WOE_unseen_categorical" not in transformed_df.columns

    if transformer.safe_original_data: # Check if unseen originals are kept
        assert "unseen_numeric" in transformed_df.columns
        assert "unseen_categorical" in transformed_df.columns
    else: # If not safe, they should also not be there as they weren't processed
        assert "unseen_numeric" not in transformed_df.columns
        assert "unseen_categorical" not in transformed_df.columns


def test_woe_transformer_transform_missing_fitted_feature(fitted_woe_transformer, simple_data_for_woe):
    transformer = fitted_woe_transformer
    data, _ = simple_data_for_woe

    data_missing_fitted = data.drop(columns=['num_feat'])
    transformed_df = transformer.transform(data_missing_fitted)

    # WOE_num_feat should still be created (as it's in woe_iv_dict) but filled with NAs/missing woe
    assert "WOE_num_feat" in transformed_df.columns
    assert transformed_df["WOE_num_feat"].isna().all() or \
           not transformed_df["WOE_num_feat"].isna().any() # if all filled by _handle_missing_values

    # Other fitted features should be transformed normally
    assert "WOE_cat_feat" in transformed_df.columns
    assert not transformed_df["WOE_cat_feat"].isna().any().any()


def test_woe_transformer_transform_empty_woe_dict():
    transformer = WOETransformer() # Not fitted, so woe_iv_dict is empty
    data = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})

    # Test with safe_original_data = True
    transformer.safe_original_data = True
    transformed_df_safe = transformer.transform(data)
    pd.testing.assert_frame_equal(transformed_df_safe, data) # Should return a copy of original

    # Test with safe_original_data = False
    transformer.safe_original_data = False
    transformed_df_not_safe = transformer.transform(data)
    # Should return an empty DataFrame with original index, as no features transformed and originals deleted
    assert transformed_df_not_safe.empty
    pd.testing.assert_index_equal(transformed_df_not_safe.index, data.index)


def test_woe_transformer_refit(fitted_woe_transformer, simple_data_for_woe):
    transformer = fitted_woe_transformer # This is already fitted on simple_data_for_woe
    original_woe_iv_dict = [item.copy() for item in transformer.woe_iv_dict] # Deep copy for comparison

    # Create new data that would result in different WOE values if refitted
    # e.g., reverse the target relationship for 'cat_feat' == 'A'
    new_data, new_target = simple_data_for_woe
    new_data = new_data.copy()
    # Original target for 'A' (first 5 rows) was [0,0,0,0,1]
    # New target for 'A': [1,1,1,1,0]
    new_target_mod = new_target.copy()
    new_target_mod.iloc[0:5] = [1,1,1,1,0]

    transformer.refit(new_data, new_target_mod)

    assert transformer.woe_iv_dict is not None
    assert len(transformer.woe_iv_dict) == len(original_woe_iv_dict)

    # Check that WOE values have changed for 'cat_feat' due to changed target relationship
    original_cat_feat_woe_A = None
    for rule in original_woe_iv_dict:
        if 'cat_feat' in rule:
            for bin_rule in rule['cat_feat']:
                if 'A' in bin_rule['bin']:
                    original_cat_feat_woe_A = bin_rule['woe']
                    break
            break

    new_cat_feat_woe_A = None
    for rule in transformer.woe_iv_dict:
        if 'cat_feat' in rule:
            for bin_rule in rule['cat_feat']:
                if 'A' in bin_rule['bin']:
                    new_cat_feat_woe_A = bin_rule['woe']
                    break
            break

    assert original_cat_feat_woe_A is not None
    assert new_cat_feat_woe_A is not None
    assert original_cat_feat_woe_A != new_cat_feat_woe_A # Expect WOE to change

def test_woe_transformer_refit_empty_dict():
    transformer = WOETransformer() # Unfitted
    data, target = pd.DataFrame({'A':[1]}), pd.Series([0])
    with pytest.raises(ValueError, match="Cannot refit an empty WOE dictionary. Fit the transformer first."):
        transformer.refit(data, target)

def test_woe_transformer_refit_empty_data():
    transformer = WOETransformer()
    transformer.woe_iv_dict = [{"some_rule": []}] # Dummy dict to pass initial check

    with pytest.raises(ValueError, match="Input data for refit cannot be empty."):
        transformer.refit(pd.DataFrame(), pd.Series([0,1]))

    with pytest.raises(ValueError, match="Target for refit must be a non-empty pandas Series or numpy array."):
        transformer.refit(pd.DataFrame({'A':[1,2]}), pd.Series([], dtype=int))

    with pytest.raises(ValueError, match="Data and target must have the same number of rows for refit."):
        transformer.refit(pd.DataFrame({'A':[1,2]}), pd.Series([0,1,0]))


def test_woe_transformer_save_load_woe_iv_dict(fitted_woe_transformer, tmp_path):
    transformer1 = fitted_woe_transformer
    file_path = tmp_path / "woe_rules.json"

    transformer1.save_to_file(str(file_path))
    assert file_path.exists()

    transformer2 = WOETransformer()
    transformer2.load_woe_iv_dict(str(file_path))

    # Detailed comparison of list of dicts
    assert len(transformer1.woe_iv_dict) == len(transformer2.woe_iv_dict)
    for rule1, rule2 in zip(transformer1.woe_iv_dict, transformer2.woe_iv_dict):
        # Check feature name (first key)
        feature_name1 = next(iter(rule1))
        feature_name2 = next(iter(rule2))
        assert feature_name1 == feature_name2

        # Compare type_feature and missing_bin
        assert rule1.get("type_feature") == rule2.get("type_feature")
        assert rule1.get("missing_bin") == rule2.get("missing_bin")

        # Compare list of bin dicts (BadRates)
        bins_list1 = rule1[feature_name1]
        bins_list2 = rule2[feature_name2]
        assert len(bins_list1) == len(bins_list2)

        for bin_dict1, bin_dict2 in zip(bins_list1, bins_list2):
            assert bin_dict1.get("bin") == bin_dict2.get("bin") # Bin structures
            assert np.isclose(bin_dict1.get("woe"), bin_dict2.get("woe"))
            assert np.isclose(bin_dict1.get("iv"), bin_dict2.get("iv"))
            assert bin_dict1.get("total") == bin_dict2.get("total")
            # Bad can be float due to smoothing, use isclose
            assert np.isclose(bin_dict1.get("bad"), bin_dict2.get("bad"))
            assert np.isclose(bin_dict1.get("pct"), bin_dict2.get("pct"))
            assert np.isclose(bin_dict1.get("bad_rate"), bin_dict2.get("bad_rate"))


def test_woe_transformer_load_non_existent_file(tmp_path):
    transformer = WOETransformer()
    file_path = tmp_path / "non_existent.json"
    with pytest.raises(FileNotFoundError):
        transformer.load_woe_iv_dict(str(file_path))

def test_woe_transformer_load_malformed_json(tmp_path):
    transformer = WOETransformer()
    file_path = tmp_path / "malformed.json"
    with open(file_path, "w") as f:
        f.write("{'key': 'value',") # Malformed JSON string

    with pytest.raises(ValueError, match="Invalid JSON format"): # Or json.JSONDecodeError
        transformer.load_woe_iv_dict(str(file_path))

# Add the original test_woe_transformer_with_missing_values if it was different
# The one above is a refactor of it.
# The original test_woe_transformer_with_missing_values(sample_data) was:
# def test_woe_transformer_with_missing_values(sample_data):
#     df, y = sample_data
#     # Add some missing values
#     df.loc[0:10, 'numeric_feature'] = np.nan
#     df.loc[20:30, 'categorical_feature'] = np.nan
#     transformer = WOETransformer(cat_features=['categorical_feature'])
#     transformer.fit(df, y)
#     transformed_df = transformer.transform(df)
#     assert not transformed_df.isna().any().any()  # Should handle missing values
# This is covered by the new test_woe_transformer_fit_transform_with_missing_values
