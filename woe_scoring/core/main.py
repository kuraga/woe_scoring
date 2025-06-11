from typing import List, Union, Optional, Any
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
            # Log: print("Warning: No features to process after prepare_data.")
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
            # Log: print("Warning: WOE dictionary is empty after processing all features.")
            pass


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using WOE encoding"""
        if not self.woe_iv_dict:
            # Log: print("Warning: WOE dictionary is empty. Returning original data or empty if no original data was to be kept.")
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
                # Log: print(f"Warning: Invalid item in woe_iv_dict: {woe_ruleset_for_feature}. Skipping.")
                continue

            # Assuming the first key is the feature name, as per binning_functions structure
            try:
                original_feature_name = next(iter(woe_ruleset_for_feature))
                # Ensure the feature name from woe_dict is actually a string (original key)
                if not isinstance(original_feature_name, str):
                    # Log: print(f"Warning: Invalid feature name key in woe_dict: {original_feature_name}. Skipping.")
                    continue
            except StopIteration: # woe_ruleset_for_feature is empty
                # Log: print(f"Warning: Empty dictionary found in woe_iv_dict. Skipping.")
                continue

            woe_bins_for_feature = woe_ruleset_for_feature.get(original_feature_name)
            # type_feature = woe_ruleset_for_feature.get("type_feature") # Used to determine cat or num
            # missing_bin_strat = woe_ruleset_for_feature.get("missing_bin")

            if not isinstance(woe_bins_for_feature, list): # Bins should be a list of dicts
                # Log: print(f"Warning: WOE info for feature '{original_feature_name}' is not a list. Skipping.")
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
                           feature_woe_rules: Dict) -> None: # feature_woe_rules includes missing_bin etc.
        """Apply WOE transformation to a single feature. woe_bins_list is list of bin dicts."""
        data_df[new_woe_col_name] = pd.NA # Initialize column

        if original_feature_name not in data_df.columns:
            # Log: print(f"Warning: Original feature '{original_feature_name}' not found in data for transform. Column '{new_woe_col_name}' will be all NA.")
            # Fill with a default WOE if applicable, or leave as NA / handle in _handle_missing_values
            # This case should ideally be caught by _handle_missing_values if all values are effectively "missing"
            # For now, let _handle_missing_values manage it.
            pass


        for bin_info_dict in woe_bins_list:
            if not isinstance(bin_info_dict, dict): continue # Skip malformed bin_info

            current_bin_values = bin_info_dict.get("bin")
            current_woe = bin_info_dict.get("woe")

            if current_bin_values is None or current_woe is None:
                # Log: print(f"Warning: Malformed bin_info {bin_info_dict} for feature {original_feature_name}. Skipping this bin.")
                continue

            # Determine if feature is categorical based on self.cat_features
            is_categorical = original_feature_name in self.cat_features

            if is_categorical:
                if not isinstance(current_bin_values, list):
                    # Log: print(f"Warning: Categorical bin for {original_feature_name} is not a list: {current_bin_values}. Skipping.")
                    continue
                # Ensure source column exists before .isin()
                if original_feature_name in data_df.columns:
                    data_df.loc[data_df[original_feature_name].isin(current_bin_values), new_woe_col_name] = current_woe
            else:
                # Assuming bin_info["bin"] is [lower_bound, upper_bound] for numerical
                # Ensure bin_info["bin"] is a list/tuple with at least 2 elements for safety,
                # though binning logic should ensure this.
                if isinstance(bin_info["bin"], (list, tuple)) and len(bin_info["bin"]) == 2:
                    lower_bound, upper_bound = bin_info["bin"][0], bin_info["bin"][1]
                # Ensure bin_info["bin"] is a list/tuple with at least 2 elements for safety
                if isinstance(current_bin_values, (list, tuple)) and len(current_bin_values) == 2:
                    lower_bound, upper_bound = current_bin_values[0], current_bin_values[1]
                    # Ensure source column exists before masking
                    if original_feature_name in data_df.columns:
                        mask = (data_df[original_feature_name] >= lower_bound) & (data_df[original_feature_name] < upper_bound)
                        data_df.loc[mask, new_woe_col_name] = current_woe
                else:
                    # Log: print(f"Warning: Unexpected bin format for numerical feature {original_feature_name}: {current_bin_values}. Skipping this bin.")
                    pass # Continue to next bin

        self._handle_missing_values(data_df, new_woe_col_name, woe_bins_list, feature_woe_rules)


    def _handle_missing_values(self, data_df: pd.DataFrame, new_woe_col_name: str,
                             woe_bins_list: List[Dict], # List of bin dicts
                             feature_woe_rules: Dict) -> None: # Full ruleset for the feature including "missing_bin"
        """Handle missing values in WOE transformation by filling NAs in the new WOE column."""
        missing_bin_strategy = feature_woe_rules.get("missing_bin")
        fill_woe_value = pd.NA # Default to NA if no strategy or rules match

        if not woe_bins_list: # No bins defined for this feature
             # Log: print(f"Warning: No WOE bins available for {new_woe_col_name}, missing values cannot be mapped via bin strategy.")
             # Keep NAs or fill with a global default if desired. For now, keeps NAs.
             return


        if missing_bin_strategy == "first":
            if woe_bins_list[0] and "woe" in woe_bins_list[0]:
                fill_woe_value = woe_bins_list[0]["woe"]
        elif missing_bin_strategy == "last":
            if woe_bins_list[-1] and "woe" in woe_bins_list[-1]:
                fill_woe_value = woe_bins_list[-1]["woe"]
        else: # Default or other strategies (e.g., if missing_bin is None or unexpected)
            # Original fallback: use WOE of first or last bin, whichever WOE is smaller.
            # This implies 'smaller WOE is safer/more common' for unassigned missings.
            try:
                woe_first = woe_bins_list[0].get("woe")
                woe_last = woe_bins_list[-1].get("woe")
                if woe_first is not None and woe_last is not None:
                    fill_woe_value = woe_first if woe_first < woe_last else woe_last
                elif woe_first is not None: # Only first bin exists or last has no WOE
                    fill_woe_value = woe_first
                elif woe_last is not None: # Only last bin exists or first has no WOE
                    fill_woe_value = woe_last
            except IndexError: # woe_bins_list might be empty
                # Log: print(f"Warning: woe_bins_list is empty for {new_woe_col_name} during missing value handling fallback.")
                pass # fill_woe_value remains pd.NA or its initial default

        if not pd.isna(fill_woe_value): # Only fill if we determined a value
            data_df[new_woe_col_name].fillna(fill_woe_value, inplace=True)
        # Else, NAs remain in new_woe_col_name if no clear strategy or WOE value found.


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
                    [b.get("bin") for b in d.get(list(d.keys())[0], []) if isinstance(b,dict)], # Get list of bins
                    d.get("type_feature"),
                    d.get("missing_bin")
                ) for d in self.woe_iv_dict if isinstance(d, dict) and d # Ensure d is a non-empty dict
            )
        except Exception as e: # Catch errors from prepare_data or Parallel execution
            # Log error: print(f"Error during WOE refitting process: {e}")
            raise RuntimeError(f"Failed to refit WOE transformations: {e}") from e


    def save_to_file(self, file_path: str) -> None:
        """Save WOE dictionary to file"""
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(self.woe_iv_dict, f, indent=4, cls=NpEncoder)
        except IOError as e:
            # Log error: print(f"IOError saving WOE dictionary to {file_path}: {e}")
            raise IOError(f"Failed to save WOE dictionary to {file_path}: {e}") from e
        except TypeError as e: # For NpEncoder issues or non-serializable content
            # Log error: print(f"TypeError during JSON serialization for WOE dictionary: {e}")
            raise TypeError(f"Failed to serialize WOE dictionary: {e}") from e


    def load_woe_iv_dict(self, file_path: str) -> None:
        """Load WOE dictionary from file"""
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                self.woe_iv_dict = json.load(f)
        except FileNotFoundError:
            # Log error: print(f"FileNotFoundError: WOE dictionary file not found at {file_path}")
            raise FileNotFoundError(f"WOE dictionary file not found: {file_path}")
        except json.JSONDecodeError as e:
            # Log error: print(f"JSONDecodeError: Error decoding WOE dictionary from {file_path}: {e}")
            raise ValueError(f"Invalid JSON format in WOE dictionary file {file_path}: {e}") from e
        except IOError as e:
            # Log error: print(f"IOError loading WOE dictionary from {file_path}: {e}")
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
                 scoring: str = "roc_auc"):

        # Store parameters, ensuring 'feature_names' is not part of model_params for FeatureSelector/Model
        temp_params = {k: v for k, v in locals().items() if k not in ['self', 'feature_names']}
        self.model_params = temp_params

        self.feature_selector: Optional[FeatureSelector] = None
        self.model: Optional[Model] = None
        self.model_results: Optional[pd.DataFrame] = None # Ensure type hint consistency

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> 'CreateModel':
        """Fit model with feature selection"""
        if 'feature_names' not in self.model_params or not self.model_params['feature_names']:
             # If feature_names are not explicitly passed to CreateModel's params,
             # they should be derived from 'data' after removing special/unused cols.
             # This assumes 'data' in fit() is the full dataset before WOE transformation.
            temp_data_for_names, feature_names_from_data = prepare_data(data.copy(), self.model_params.get('special_cols'))
            if self.model_params.get('unused_cols'):
                feature_names_from_data = [fn for fn in feature_names_from_data if fn not in self.model_params['unused_cols']]
            current_model_params = {**self.model_params, 'feature_names': feature_names_from_data}
        else:
            current_model_params = self.model_params

        try:
            self.feature_selector = FeatureSelector(**current_model_params)
            # Ensure data passed to select contains only feature_names expected by selector
            # This usually means data after WOE transformation if selector runs on WOE features.
            # However, selector is often run on original or binned data, not WOE values directly.
            # Assuming 'data' here is the data on which selection should happen.
            # And feature_names in current_model_params are columns in this 'data'.

            # The 'feature_names' argument to .select() should be the list of features to select from.
            # If data is already WOE transformed, then feature_names should be WOE columns.
            # This part is tricky: CreateModel.fit is typically called on WOE transformed data.
            # So, data.columns should be the WOE features.
            # model_params['feature_names'] in FeatureSelector likely refers to original features for Gini/IV.
            # This needs careful review of FeatureSelector's internal needs.
            # For now, assume data.columns are the features to select from if model_params['feature_names'] is not set.

            features_to_select_from = current_model_params['feature_names']
            if not features_to_select_from and isinstance(data, pd.DataFrame):
                features_to_select_from = list(data.columns)

            selected_features = self.feature_selector.select(data, target, features_to_select_from)
            if not selected_features:
                raise ValueError("Feature selection returned no features. Cannot fit model.")

        except Exception as e:
            # Log error: print(f"Error during feature selection: {e}")
            raise RuntimeError(f"Feature selection failed: {e}") from e

        try:
            self.model = Model(**current_model_params) # Pass all relevant params
            self.model.get_model(data[selected_features], target) # Fit model on selected features
        except Exception as e:
            # Log error: print(f"Error during model fitting: {e}")
            raise RuntimeError(f"Model fitting failed: {e}") from e

        try:
            self.model_results = _calc_model_results(self.model)
        except Exception as e:
            # Log error: print(f"Error calculating model results: {e}")
            # self.model_results might remain None or be an empty DataFrame
            self.model_results = pd.DataFrame() # Ensure it's a DF even if empty
            # Optionally re-raise if model_results are critical for subsequent steps
            # raise RuntimeError(f"Failed to calculate model results: {e}") from e
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        try:
            return self.model.predict_proba(data)
        except Exception as e:
            # Log error: print(f"Error during predict_proba: {e}")
            raise RuntimeError(f"Prediction (proba) failed: {e}") from e

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        try:
            return self.model.predict(data)
        except Exception as e:
            # Log error: print(f"Error during predict: {e}")
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
