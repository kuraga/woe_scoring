from typing import List, Union, Optional, Any, Dict
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from joblib import Parallel, delayed

from .binning.functions import (cat_processing, find_cat_features,
                              num_processing, prepare_data, refit)
# Updated imports for model functions
from .model.functions import (calc_model_results as _calc_model_results, # Still from functions.py
                              save_scorecard as _save_scorecard)       # Still from functions.py
from .model import model_analyzer # New module for save_reports
from .model import sql_generator  # New module for generate_sql

from .model.model import Model
import logging # Add logging import

from .model.selector import FeatureSelector


class NpEncoder(json.JSONEncoder):
    """JSON encoder for NumPy types"""
    def default(self, o):
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class WOETransformer(BaseEstimator, TransformerMixin):
    """Weight of Evidence (WOE) transformer"""

    def __init__(
            self,
            max_bins: Union[int, float] = 10,
            min_pct_group: float = 0.05,
            n_jobs: int = 1,
            prefix: str = "WOE_",
            merge_type: str = "chi2",
            cat_features: Optional[List[str]] = None,
            special_cols: Optional[List[str]] = None,
            cat_features_threshold: int = 0,
            diff_woe_threshold: float = 0.05,
            safe_original_data: bool = False,
    ):
        self.max_bins = max_bins
        self.min_pct_group = min_pct_group
        self.n_jobs = n_jobs
        self.prefix = prefix
        self.merge_type = merge_type
        self.cat_features = cat_features if cat_features is not None else []
        self.special_cols = special_cols if special_cols is not None else []
        self.cat_features_threshold = cat_features_threshold
        self.diff_woe_threshold = diff_woe_threshold
        self.safe_original_data = safe_original_data

        self.classes_: Optional[np.ndarray] = None
        self.woe_iv_dict: List = [] # List of dictionaries
        self.feature_names: List[str] = []
        self.num_features: List[str] = []
        self.logger = logging.getLogger(__name__) # Initialize logger

        # Basic parameter validation
        if not (0 < min_pct_group < 1):
            raise ValueError("min_pct_group must be between 0 and 1.")
        if isinstance(max_bins, int) and max_bins <= 0:
            raise ValueError("max_bins (if int) must be positive.")
        if isinstance(max_bins, float) and not (0 < max_bins <= 1):
             raise ValueError("max_bins (if float) must be between 0 and 1.")


    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> 'WOETransformer':
        """Fit WOE transformer to data"""
        if data.empty:
            raise ValueError("Input data DataFrame cannot be empty.")
        if not isinstance(target, (pd.Series, np.ndarray)) or len(target) == 0:
            raise ValueError("Target must be a non-empty pandas Series or numpy array.")
        if len(data) != len(target):
            raise ValueError("Data and target must have the same number of rows.")

        try:
            processed_data, self.feature_names = prepare_data(data.copy(), self.special_cols) # Work on a copy
        except TypeError as e:
            raise TypeError(f"Error in prepare_data: {e}") from e

        if not self.feature_names:
            self.logger.warning("No features to process after prepare_data.")
            self.woe_iv_dict = []
            return self

        self.classes_ = unique_labels(target)

        if not self.cat_features and self.cat_features_threshold > 0:
            self.cat_features = find_cat_features(
                x=data,
                feature_names=self.feature_names,
                cat_features_threshold=self.cat_features_threshold
            )

        self._process_features(data, target)
        return self

    def _process_features(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> None:
        """Process features in parallel"""
        if self.cat_features:
            self.num_features = [f for f in self.feature_names if f not in self.cat_features]
            cat_results = Parallel(n_jobs=self.n_jobs)(
                delayed(cat_processing)(
                    data[col], target,
                    self.min_pct_group,
                    self.max_bins,
                    self.diff_woe_threshold
                ) for col in self.cat_features
            )
            try:
                self.woe_iv_dict.extend(cat_results)
            except RuntimeError as e: # Catch errors from cat_processing
                raise RuntimeError(f"Error processing categorical features: {e}") from e
        else:
            self.num_features = list(self.feature_names) # Ensure it's a list copy

        if self.num_features: # Only run if there are numerical features
            try:
                num_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(num_processing)(
                        data[col], target,
                        self.min_pct_group,
                        self.max_bins,
                        self.diff_woe_threshold,
                        self.merge_type
                    ) for col in self.num_features
                )
                self.woe_iv_dict.extend(num_results)
            except RuntimeError as e: # Catch errors from num_processing
                raise RuntimeError(f"Error processing numerical features: {e}") from e

        if not self.woe_iv_dict:
            self.logger.warning("WOE dictionary is empty after processing all features.")
            pass


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using WOE encoding"""
        if not self.woe_iv_dict:
            self.logger.warning("WOE dictionary is empty. Returning original data or empty if no original data was to be kept.")
            # Depending on safe_original_data, this might return an empty DataFrame if no features are transformed.
            if self.safe_original_data:
                return data.copy()
            else: # Return empty DataFrame with no columns if no features were transformed and originals deleted.
                  # Or, more robustly, return only special_cols if they exist and were meant to be kept.
                  # For now, this will lead to an empty DF if transformed_features is empty.
                  # A better approach might be to not delete if no transformations occurred.
                return pd.DataFrame(index=data.index)


        result = data.copy()
        transformed_features = []
        processed_original_features = set()

        for woe_ruleset_for_feature in self.woe_iv_dict:
            if not isinstance(woe_ruleset_for_feature, dict) or not woe_ruleset_for_feature:
                self.logger.warning(f"Invalid item in woe_iv_dict: {woe_ruleset_for_feature}. Skipping.")
                continue

            # Assuming the first key is the feature name, as per binning_functions structure
            try:
                original_feature_name = next(iter(woe_ruleset_for_feature))
                # Ensure the feature name from woe_dict is actually a string (original key)
                if not isinstance(original_feature_name, str):
                    self.logger.warning(f"Invalid feature name key in woe_dict: {original_feature_name}. Skipping.")
                    continue
            except StopIteration: # woe_ruleset_for_feature is empty
                self.logger.warning("Empty dictionary found in woe_iv_dict. Skipping.")
                continue

            woe_bins_for_feature = woe_ruleset_for_feature.get(original_feature_name)
            # type_feature = woe_ruleset_for_feature.get("type_feature") # Used to determine cat or num
            # missing_bin_strat = woe_ruleset_for_feature.get("missing_bin")

            if not isinstance(woe_bins_for_feature, list): # Bins should be a list of dicts
                self.logger.warning(f"WOE info for feature '{original_feature_name}' is not a list. Skipping.")
                continue

            new_woe_col_name = f"{self.prefix}{original_feature_name}"
            transformed_features.append(new_woe_col_name)
            processed_original_features.add(original_feature_name)

            self._apply_woe_transform(result, original_feature_name, new_woe_col_name,
                                      woe_bins_for_feature, woe_ruleset_for_feature)

        # Delete original features if safe_original_data is False
        if not self.safe_original_data:
            for feature_to_delete in processed_original_features:
                if feature_to_delete in result.columns:
                    del result[feature_to_delete]

        # Return only transformed features if safe_original_data is False and there are transformed features
        # Otherwise, return the result df which might contain original + transformed
        if not self.safe_original_data and transformed_features:
            # Also include any special_cols if they were intended to be kept implicitly
            cols_to_return = transformed_features
            if self.special_cols:
                cols_to_return = list(set(cols_to_return + [sc for sc in self.special_cols if sc in result.columns]))
            return result[cols_to_return]
        else: # safe_original_data is True OR no transformed_features (e.g. if woe_iv_dict was empty)
            return result


    def _apply_woe_transform(self, data_df: pd.DataFrame, original_feature_name: str,
                           new_woe_col_name: str, woe_bins_list: List[Dict],
                           feature_woe_rules: Dict) -> None:
        """Apply WOE transformation to a single feature. woe_bins_list is list of bin dicts (BadRates)."""
        data_df[new_woe_col_name] = pd.NA # Initialize column

        if original_feature_name not in data_df.columns:
            self.logger.warning(f"Original feature '{original_feature_name}' not found in data for transform. Column '{new_woe_col_name}' will be all NA.")
            # All values will be NA, and _handle_missing_values will take care of them.
            pass # No explicit filling here, _handle_missing_values will do it.

        is_categorical = original_feature_name in self.cat_features
        for bin_info_item in woe_bins_list: # woe_bins_list contains BadRates objects or dicts
            # Pass bin_info_item directly to _apply_single_bin_woe which now handles both types
            self._apply_single_bin_woe(data_df, original_feature_name, new_woe_col_name,
                                      bin_info_item, is_categorical)

        self._handle_missing_values(data_df, original_feature_name, new_woe_col_name, woe_bins_list, feature_woe_rules)


    def _apply_single_bin_woe(self, data_df: pd.DataFrame, original_feature_name: str, new_woe_col_name: str,
                                bin_info: Dict, is_categorical: bool) -> None:
        """Applies WOE for a single bin to the DataFrame."""
        # Get bin values and WOE, handling both dictionary and BadRates object
        if hasattr(bin_info, 'bin') and hasattr(bin_info, 'woe'):  # It's a BadRates object
            current_bin_values = bin_info.bin
            current_woe = bin_info.woe
        else:  # It's a dictionary
            current_bin_values = bin_info.get("bin")
            current_woe = bin_info.get("woe")

        if current_bin_values is None or current_woe is None:
            self.logger.warning(f"Malformed bin_info {bin_info} for feature {original_feature_name}. Skipping this bin.")
            return

        if is_categorical:
            if not isinstance(current_bin_values, list):
                self.logger.warning(f"Categorical bin for {original_feature_name} is not a list: {current_bin_values}. Skipping.")
                return
            if original_feature_name in data_df.columns: # Ensure source column exists
                # Ensure all items in current_bin_values are comparable to data_df column
                # This might involve type conversion if data_df column is numeric but bin_values are strings from JSON
                # For now, assume types are compatible as per prior processing.
                try:
                    mask = data_df[original_feature_name].isin(current_bin_values)
                    data_df.loc[mask, new_woe_col_name] = current_woe
                except TypeError as e:
                    self.logger.error(f"TypeError during .isin() for categorical feature {original_feature_name} with bin {current_bin_values}. Data type: {data_df[original_feature_name].dtype}. Error: {e}")
        else: # Numerical
            if isinstance(current_bin_values, (list, tuple)) and len(current_bin_values) == 2:
                lower_bound, upper_bound = current_bin_values[0], current_bin_values[1]
                if original_feature_name in data_df.columns: # Ensure source column exists
                    try:
                        # Ensure bounds are numeric if column is numeric
                        # This can be an issue if bounds are NINF/INF as strings from JSON
                        # For now, assume they are loaded correctly as numbers/np.inf
                        mask = (data_df[original_feature_name] >= lower_bound) & \
                               (data_df[original_feature_name] < upper_bound)
                        data_df.loc[mask, new_woe_col_name] = current_woe
                    except TypeError as e:
                         self.logger.error(f"TypeError during numerical comparison for {original_feature_name} with bounds {lower_bound}, {upper_bound}. Data type: {data_df[original_feature_name].dtype}. Error: {e}")
            else:
                self.logger.warning(f"Unexpected bin format for numerical feature {original_feature_name}: {current_bin_values}. Skipping this bin.")


    def _handle_missing_values(self, data_df: pd.DataFrame, original_feature_name: str, new_woe_col_name: str,
                             woe_bins_list: List[Dict],
                             feature_woe_rules: Dict) -> None:
        """Handle missing values in WOE transformation by filling NAs in the new WOE column."""
        # This method now also handles cases where the original feature might be missing in the input data_df
        # by checking data_df[new_woe_col_name].isna().sum() > 0 after bin applications.

        missing_bin_strategy = feature_woe_rules.get("missing_bin")
        fill_woe_value = pd.NA

        # Determine the WOE value to use for missing/unmapped values
        if not woe_bins_list:
            self.logger.warning(f"No WOE bins available for {new_woe_col_name} (feature: {original_feature_name}), "
                                f"missing/unmapped values cannot be mapped via bin strategy.")
        elif missing_bin_strategy == "first":
            if woe_bins_list[0]:
                # Handle both dict and BadRates object
                if hasattr(woe_bins_list[0], 'woe'):
                    fill_woe_value = woe_bins_list[0].woe
                elif isinstance(woe_bins_list[0], dict) and "woe" in woe_bins_list[0]:
                    fill_woe_value = woe_bins_list[0]["woe"]
                else:
                    self.logger.warning(f"Missing strategy 'first' for {new_woe_col_name} but first bin has no WOE.")
            else:
                self.logger.warning(f"Missing strategy 'first' for {new_woe_col_name} but first bin is invalid.")
        elif missing_bin_strategy == "last":
            if woe_bins_list[-1]:
                # Handle both dict and BadRates object
                if hasattr(woe_bins_list[-1], 'woe'):
                    fill_woe_value = woe_bins_list[-1].woe
                elif isinstance(woe_bins_list[-1], dict) and "woe" in woe_bins_list[-1]:
                    fill_woe_value = woe_bins_list[-1]["woe"]
                else:
                    self.logger.warning(f"Missing strategy 'last' for {new_woe_col_name} but last bin has no WOE.")
            else:
                self.logger.warning(f"Missing strategy 'last' for {new_woe_col_name} but last bin is invalid.")
        else: # Default fallback logic (missing_bin_strategy is None or unexpected)
            self.logger.info(f"No specific missing_bin_strategy for {new_woe_col_name} (feature: {original_feature_name}) or strategy is '{missing_bin_strategy}'. "
                             "Using fallback: smaller WOE of first/last bin if available.")
            try:
                # Get first and last WOE values, handling both dict and BadRates object
                if woe_bins_list:
                    if hasattr(woe_bins_list[0], 'woe'):
                        woe_first = woe_bins_list[0].woe
                    else:
                        woe_first = woe_bins_list[0].get("woe") if isinstance(woe_bins_list[0], dict) else None

                    if hasattr(woe_bins_list[-1], 'woe'):
                        woe_last = woe_bins_list[-1].woe
                    else:
                        woe_last = woe_bins_list[-1].get("woe") if isinstance(woe_bins_list[-1], dict) else None
                else:
                    woe_first = None
                    woe_last = None

                if woe_first is not None and woe_last is not None:
                    fill_woe_value = woe_first if woe_first < woe_last else woe_last
                elif woe_first is not None:
                    fill_woe_value = woe_first
                elif woe_last is not None:
                    fill_woe_value = woe_last
                else: # Neither first nor last bin has a WOE value
                    self.logger.warning(f"Fallback for {new_woe_col_name} could not find WOE in first or last bin.")
            except IndexError: # Should be caught by `if not woe_bins_list:` but as a safeguard
                self.logger.warning(f"Fallback for {new_woe_col_name} failed due to IndexError (empty woe_bins_list).")

        # Apply the determined fill_woe_value to any remaining NAs in the WOE column
        # This includes actual NaNs in original_feature_name, or values not covered by any bin,
        # or if original_feature_name was missing entirely.
        if data_df[new_woe_col_name].isna().any():
            if not pd.isna(fill_woe_value):
                self.logger.info(f"Filling {data_df[new_woe_col_name].isna().sum()} NA values in '{new_woe_col_name}' with WOE: {fill_woe_value:.4f} (Strategy: {missing_bin_strategy if missing_bin_strategy else 'fallback/default'})")
                data_df[new_woe_col_name].fillna(fill_woe_value, inplace=True)
            else:
                self.logger.warning(f"{data_df[new_woe_col_name].isna().sum()} NA values remain in '{new_woe_col_name}' as no valid fill WOE was determined (Strategy: {missing_bin_strategy if missing_bin_strategy else 'fallback/default'}).")


    def refit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> None:
        """Refit WOE transformer with new data"""
        if data.empty:
            raise ValueError("Input data for refit cannot be empty.")
        if not isinstance(target, (pd.Series, np.ndarray)) or len(target) == 0:
            raise ValueError("Target for refit must be a non-empty pandas Series or numpy array.")
        if len(data) != len(target):
            raise ValueError("Data and target must have the same number of rows for refit.")
        if not self.woe_iv_dict:
             raise ValueError("Cannot refit an empty WOE dictionary. Fit the transformer first.")

        try:
            processed_data, self.feature_names = prepare_data(data.copy(), self.special_cols)

            self.woe_iv_dict = Parallel(n_jobs=self.n_jobs)(
                delayed(refit)(
                    processed_data[list(d.keys())[0]], # Assumes d.keys()[0] is the feature name
                    target,
                    [b.get("bin") for b in d.get(list(d.keys())[0], []) if isinstance(b, dict)], # Get list of bins
                    d.get("type_feature"),
                    d.get("missing_bin")
                ) for d in self.woe_iv_dict if isinstance(d, dict) and d # Ensure d is a non-empty dict
            )
        except Exception as e: # Catch errors from prepare_data or Parallel execution
            self.logger.error(f"Error during WOE refitting process: {e}")
            raise RuntimeError(f"Failed to refit WOE transformations: {e}") from e


    def save_to_file(self, file_path: str) -> None:
        """Save WOE dictionary to file"""
        try:
            # Make sure we have a serializable dictionary (no BadRates objects)
            serializable_woe_dict = []
            for feature_dict in self.woe_iv_dict:
                if not isinstance(feature_dict, dict):
                    self.logger.warning(f"Skipping non-dict item in woe_iv_dict: {feature_dict}")
                    continue

                # Create a deep copy to avoid modifying original
                feature_dict_copy = {}
                for key, value in feature_dict.items():
                    if key in ["missing_bin", "type_feature"]:
                        feature_dict_copy[key] = value
                    else:
                        # This is the feature name key with list of bin dicts
                        feature_dict_copy[key] = []
                        for bin_item in value:
                            if hasattr(bin_item, '__dict__'):  # It's a BadRates object
                                feature_dict_copy[key].append({
                                    "bin": bin_item.bin,
                                    "total": bin_item.total,
                                    "bad": bin_item.bad,
                                    "pct": bin_item.pct,
                                    "bad_rate": bin_item.bad_rate,
                                    "woe": bin_item.woe,
                                    "iv": bin_item.iv
                                })
                            else:  # It's already a dict
                                feature_dict_copy[key].append(bin_item)

                serializable_woe_dict.append(feature_dict_copy)

            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(serializable_woe_dict, f, indent=4, cls=NpEncoder)
        except IOError as e:
            self.logger.error(f"IOError saving WOE dictionary to {file_path}: {e}")
            raise IOError(f"Failed to save WOE dictionary to {file_path}: {e}") from e
        except TypeError as e: # For NpEncoder issues or non-serializable content
            self.logger.error(f"TypeError during JSON serialization for WOE dictionary: {e}")
            raise TypeError(f"Failed to serialize WOE dictionary: {e}") from e


    def load_woe_iv_dict(self, file_path: str) -> None:
        """Load WOE dictionary from file"""
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                self.woe_iv_dict = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"FileNotFoundError: WOE dictionary file not found at {file_path}")
            raise FileNotFoundError(f"WOE dictionary file not found: {file_path}")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSONDecodeError: Error decoding WOE dictionary from {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in WOE dictionary file {file_path}: {e}") from e
        except IOError as e:
            self.logger.error(f"IOError loading WOE dictionary from {file_path}: {e}")
            raise IOError(f"Failed to load WOE dictionary from {file_path}: {e}") from e


class CreateModel(BaseEstimator, TransformerMixin):
    """Model creation and feature selection class"""

    def __init__(self, selection_method: str = 'rfe',
                 model_type: str = 'sklearn',
                 max_vars: Optional[Union[int, float]] = None,
                 special_cols: Optional[List[str]] = None,
                 unused_cols: Optional[List[str]] = None,
                 n_jobs: int = 1,
                 gini_threshold: float = 5.0,
                 iv_threshold: float = 0.05,
                 corr_threshold: float = 0.5,
                 min_pct_group: float = 0.05,
                 random_state: Optional[int] = None,
                 class_weight: str = 'balanced',
                 direction: str = "forward",
                 cv: int = 3,
                 l1_exp_scale: int = 4,
                 l1_grid_size: int = 20,
                 scoring: str = "roc_auc",
                 # Add woe_rules to init, needed for FeatureSelector if it calculates IV
                 woe_transformer_rules: Optional[List[Dict]] = None,
                 feature_names: Optional[List[str]] = None): # Original feature names, not WOE transformed

        self.selection_method = selection_method
        self.model_type = model_type
        self.max_vars = max_vars
        self.special_cols = special_cols if special_cols is not None else []
        self.unused_cols = unused_cols if unused_cols is not None else []
        self.n_jobs = n_jobs
        self.gini_threshold = gini_threshold
        self.iv_threshold = iv_threshold
        self.corr_threshold = corr_threshold
        self.min_pct_group = min_pct_group # Used by FeatureSelector if it re-bins for Gini/IV
        self.random_state = random_state
        self.class_weight = class_weight
        self.direction = direction
        self.cv = cv
        self.l1_exp_scale = l1_exp_scale
        self.l1_grid_size = l1_grid_size
        self.scoring = scoring
        self.woe_transformer_rules = woe_transformer_rules if woe_transformer_rules is not None else []
        self.input_feature_names = feature_names if feature_names is not None else [] # Original (non-WOE) names

        self.logger = logging.getLogger(__name__)
        self.feature_selector: Optional[FeatureSelector] = None
        self.model: Optional[Model] = None
        self.model_results: Optional[pd.DataFrame] = None

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> 'CreateModel':
        """Fit model with feature selection.
        'data' is expected to be WOE-transformed data.
        'self.input_feature_names' should be original feature names if provided,
        or derived from woe_transformer_rules.
        """
        self.logger.info("Starting CreateModel fitting process.")

        original_feature_names_for_selector: List[str] = []
        if self.input_feature_names:
            original_feature_names_for_selector = self.input_feature_names
        elif self.woe_transformer_rules:
            self.logger.info("Deriving original feature names from woe_transformer_rules for FeatureSelector.")
            for rule_set in self.woe_transformer_rules:
                if isinstance(rule_set, dict) and rule_set:
                    original_feature_names_for_selector.append(next(iter(rule_set)))

        if not original_feature_names_for_selector and self.woe_transformer_rules is not None :
             self.logger.warning("Could not derive original feature names for FeatureSelector. "
                                 "IV/Gini based selection might be impacted if selector needs them.")

        # features_in_woe_data are the columns in the input 'data' (WOE names)
        features_in_woe_data = list(data.columns)

        selector_params = {
            "selection_method": self.selection_method, "model_type": self.model_type,
            "max_vars": self.max_vars, "special_cols": self.special_cols, # special_cols might be irrelevant if data is already WOE
            "unused_cols": self.unused_cols, # unused_cols might be irrelevant here
            "n_jobs": self.n_jobs, "gini_threshold": self.gini_threshold,
            "iv_threshold": self.iv_threshold, "corr_threshold": self.corr_threshold,
            "min_pct_group": self.min_pct_group, "random_state": self.random_state,
            "direction": self.direction, "cv": self.cv, "scoring": self.scoring,
            "feature_names": original_feature_names_for_selector, # Original names
            "woe_rules": self.woe_transformer_rules # Pass WOE rules for IV calculation
        }

        model_constructor_params = {
            "model_type": self.model_type, "random_state": self.random_state,
            "class_weight": self.class_weight, "n_jobs": self.n_jobs, "cv": self.cv,
            "l1_exp_scale": self.l1_exp_scale, "l1_grid_size": self.l1_grid_size,
            "scoring": self.scoring
        }

        try:
            self.feature_selector = FeatureSelector(**selector_params) # type: ignore
            # FeatureSelector.select expects WOE data, and list of WOE features to select from.
            # It uses its internal feature_names (original) and woe_rules to map for IV if needed.
            self.logger.info(f"Starting feature selection using method: {self.selection_method}")
            selected_woe_features = self.feature_selector.select(data, target, features_in_woe_data)

            if not selected_woe_features:
                self.logger.error("Feature selection returned no features. Cannot fit model.")
                raise ValueError("Feature selection returned no features. Cannot fit model.")
            self.logger.info(f"Selected {len(selected_woe_features)} features: {selected_woe_features}")

        except Exception as e:
            self.logger.error(f"Error during feature selection: {e}", exc_info=True)
            raise RuntimeError(f"Feature selection failed: {e}") from e

        try:
            self.model = Model(**model_constructor_params) # type: ignore
            self.logger.info(f"Fitting model of type: {self.model_type} on selected features.")
            self.model.get_model(data[selected_woe_features], target)
        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}", exc_info=True)
            raise RuntimeError(f"Model fitting failed: {e}") from e

        try:
            self.logger.info("Calculating model results.")
            self.model_results = _calc_model_results(self.model)
        except Exception as e:
            self.logger.error(f"Error calculating model results: {e}", exc_info=True)
            self.model_results = pd.DataFrame()
        self.logger.info("CreateModel fitting process completed.")
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        try:
            return self.model.predict_proba(data)
        except Exception as e:
            # print(f"Error during predict_proba: {e}") # Log this if CreateModel had a logger
            raise RuntimeError(f"Prediction (proba) failed: {e}") from e

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        try:
            return self.model.predict(data)
        except Exception as e:
            # print(f"Error during predict: {e}") # Log this if CreateModel had a logger
            raise RuntimeError(f"Prediction failed: {e}") from e

    def save_reports(self, path: str) -> None:
        """Save model reports"""
        if self.model is None:
            raise ValueError("Model must be fitted before saving reports")
        if self.model.model is None: # self.model.model is the actual statsmodels object
             raise ValueError("Actual statsmodel (self.model.model) not found. Cannot save reports.")

        # Assuming self.model (Model class instance) has methods to get these:
        # These methods would need to be implemented in the Model class.
        # For example:
        # def get_summary_text(self): return self.model.summary().as_text()
        # def get_wald_test_summary_df(self): return self.model.wald_test_terms().summary_frame()
        try:
            # self.model is an instance of Model class from model.py
            # self.model.model is an instance of SMWrapper (if statsmodels) or LogisticRegressionCV
            # self.model.model.model_ is the actual statsmodels results object from SMWrapper
            actual_sm_model_obj = self.model.model.model_
            summary_text = actual_sm_model_obj.summary().as_text()
            wald_df = actual_sm_model_obj.wald_test_terms().summary_frame()
        except AttributeError:
            # This might also occur if the model is sklearn, which doesn't have .summary() etc.
            # The original _save_reports was also only for statsmodels. We should check model type.
            if self.model.model_type == 'statsmodels':
                raise AttributeError("The 'model' object (SMWrapper) or its underlying statsmodels model "
                                     "is not as expected or lacks summary/wald_test_terms methods.")
            else:
                # For sklearn models, saving these specific reports might not be applicable.
                # Consider logging a message or skipping.
                print(f"Skipping saving statsmodels-specific reports for model type: {self.model.model_type}")
                return


        model_analyzer.save_model_reports(
            model_summary_text=summary_text,
            wald_test_summary_df=wald_df,
            path=path
        )

    def generate_sql(self, encoder: Any) -> str: # Added type hint for encoder
        """Generate SQL for model scoring"""
        if self.model is None:
            raise ValueError("Model must be fitted before generating SQL")
        # Parameter names in generate_sql_query are:
        # woe_encoder_info, woe_feature_names, model_coefficients, model_intercept
        return sql_generator.generate_sql_query(
                           woe_encoder_info=encoder, # Pass encoder (expected to be list of woe_iv_dicts)
                           woe_feature_names=self.model.feature_names_,
                           model_coefficients=self.model.coef_,
                           model_intercept=self.model.intercept_
                           )

    def save_scorecard(self, encoder: Any, path: str = '.', # Added type hint for encoder
                      base_scorecard_points: int = 444,
                      odds: int = 10,
                      points_to_double_odds: int = 69) -> None:
        """Save scorecard to file"""
        if self.model is None or self.model_results is None:
            raise ValueError("Model must be fitted and results calculated before saving scorecard")

        # Parameter names in the refactored _save_scorecard are:
        # woe_feature_names_in_model, woe_encoder_rules, model_coeff_pvalue_df,
        # base_points, odds, points_to_double_odds, output_path
        _save_scorecard(woe_feature_names_in_model=self.model.feature_names_,
                       woe_encoder_rules=encoder, # Pass encoder (expected to be list of woe_iv_dicts)
                       model_coeff_pvalue_df=self.model_results,
                       base_points=base_scorecard_points,
                       odds=odds,
                       points_to_double_odds=points_to_double_odds,
                       output_path=path
                       )
