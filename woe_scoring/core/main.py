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
        """
        Transforms the input DataFrame using the Weight of Evidence (WOE) technique.

        Args:
            data (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """

        data = data.copy()
        features_to_delete = []

        # Pre-create all new feature columns
        for woe_iv in self.woe_iv_dict:
            feature = list(woe_iv)[0]
            new_feature = self.prefix + feature
            data[new_feature] = np.nan
            if not self.safe_original_data:
                features_to_delete.append(feature)

        # Apply transformations
        for woe_iv in self.woe_iv_dict:
            feature = list(woe_iv)[0]
            woe_iv_feature = woe_iv[feature]
            new_feature = self.prefix + feature

            # Apply bins based on feature type
            if feature in self.cat_features:
                # Categorical features - vectorized approach using map with default to NaN
                bin_map = {}
                for bin_values in woe_iv_feature:
                    for bin_val in bin_values["bin"]:
                        bin_map[bin_val] = bin_values["woe"]

                # Convert to category first for efficiency with large datasets
                data.loc[:, new_feature] = data[feature].map(bin_map)
            else:
                # Numerical features
                for bin_values in woe_iv_feature:
                    mask = np.logical_and(
                        data[feature] >= np.min(bin_values["bin"]),
                        data[feature] < np.max(bin_values["bin"])
                    )
                    data.loc[mask, new_feature] = bin_values["woe"]

            # Handle missing values efficiently
            missing_bin = woe_iv["missing_bin"]
            missing_value = (
                woe_iv_feature[0]["woe"] if missing_bin == "first" or
                (missing_bin is None and woe_iv_feature[0]["woe"] < woe_iv_feature[-1]["woe"])
                else woe_iv_feature[-1]["woe"]
            )
            data.loc[data[new_feature].isna(), new_feature] = missing_value

        # Remove original features if needed
        if features_to_delete:
            data = data.drop(columns=features_to_delete)

        return data

    def save_to_file(self, file_path: str) -> None:
        """
        Save the woe_iv_dict to a JSON file at the specified file path.

        Args:
            file_path (str): The path where the file should be saved.


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


        data, self.feature_names = prepare_data(data=data, special_cols=self.special_cols)

        # Ensure target is numpy array for consistency
        target_values = target.values if hasattr(target, 'values') else np.array(target)

        # Process in parallel with optimized parameters
        self.woe_iv_dict = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(refit)(
                data[list(woe_iv.keys())[0]],
                target_values,
                [_bin["bin"] for _bin in woe_iv[list(woe_iv.keys())[0]]],
                woe_iv["type_feature"],
                woe_iv["missing_bin"]
            ) for woe_iv in self.woe_iv_dict
        )


class CreateModel(BaseEstimator, TransformerMixin):
    """
    Class to create a predictive model with automatic feature selection.

    This class automates the feature selection process and model training,
    supporting multiple selection techniques and model types.

    Args:
        selection_method (str): Feature selection method: 'rfe', 'sfs', or 'iv'.
            - 'rfe': Recursive Feature Elimination
            - 'sfs': Sequential Feature Selection
            - 'iv': Information Value based selection
        model_type (str): Model type: 'sklearn' or 'statsmodel'.
        max_vars (int, float, None): Maximum number of features to select.
            If float < 1, interpreted as a percentage of total features.
            If None, no limit is applied.
        special_cols (list, optional): Special columns to include in selection.
        unused_cols (list, optional): Columns to exclude from selection.
        n_jobs (int): Number of CPU cores for parallelization.
        gini_threshold (float): Minimum Gini score to retain a feature.
        iv_threshold (float): Minimum information value threshold for 'iv' method.
        corr_threshold (float): Maximum correlation allowed between features.
        min_pct_group (float): Minimum percentage for each target class.
        random_state (int, optional): Random seed for reproducible results.
        class_weight (str): Class weight strategy ('balanced' or None).
        direction (str): Feature selection direction: 'forward' or 'backward'.
        cv (int): Number of cross-validation folds.
        l1_exp_scale (int): Exponent scale for L1 regularization grid.
        l1_grid_size (int): Grid size for L1 regularization search.
        scoring (str): Metric for model evaluation (e.g., 'roc_auc').
    """
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
        Fit the model with the given data and target.

        Args:
            data (pd.DataFrame): The input data.
            target (Union[pd.Series, np.ndarray]): The target values.

        Returns:
            The fitted model.
        """
        # Prepare data and filter features
        data, self.feature_names_ = prepare_data(data=data, special_cols=self.special_cols)

        # Remove unused columns if specified
        if self.unused_cols:
            self.feature_names_ = [f for f in self.feature_names_ if f not in self.unused_cols]

        # Calculate max_vars if it's a ratio
        if self.max_vars is not None and self.max_vars < 1:
            self.max_vars = int(len(self.feature_names_) * self.max_vars)

        # Filter features based on minimum group percentage
        self.feature_names_ = check_min_pct_group(
            data=data, feature_names=self.feature_names_, min_pct_group=self.min_pct_group
        )

        # Calculate Gini scores for all features in parallel
        self.features_gini_scores = calc_features_gini_quality(
            data=data,
            target=target,
            feature_names=self.feature_names_,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            cv=self.cv,
            scoring=self.scoring
        )

        # Filter features by Gini threshold
        self.feature_names_ = check_features_gini_threshold(
            feature_names=self.feature_names_,
            features_gini_scores=self.features_gini_scores,
            gini_threshold=self.gini_threshold
        )

        # Create feature selector
        feature_selector = FeatureSelector(
            selection_type=self.selection_method,
            max_vars=self.max_vars,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            class_weight=self.class_weight,
            direction=self.direction,
            cv=self.cv,
            l1_exp_scale=self.l1_exp_scale,
            l1_grid_size=self.l1_grid_size,
            scoring=self.scoring,
            iv_threshold=self.iv_threshold
        )

        # Select initial features and check for correlations
        selected_features = feature_selector.select(data, target, self.feature_names_)
        selected_features = check_correlation_threshold(
            data=data,
            feature_names=selected_features,
            features_gini_scores=self.features_gini_scores,
            corr_threshold=self.corr_threshold,
        )

        # Initialize model
        selected_model = Model(
            model_type=self.model_type,
            n_jobs=self.n_jobs,
            l1_exp_scale=self.l1_exp_scale,
            l1_grid_size=self.l1_grid_size,
            cv=self.cv,
            class_weight=self.class_weight,
            random_state=self.random_state,
            scoring=self.scoring
        )

        # Iteratively improve model by removing bad features
        self.model = selected_model.get_model(data[selected_features], target)
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            bad_features = find_bad_features(selected_model)
            if not bad_features:
                break

            # Remove bad features and reselect
            self.feature_names_ = [f for f in self.feature_names_ if f not in bad_features]
            if not self.feature_names_:  # Prevent empty feature list
                break

            selected_features = feature_selector.select(data, target, self.feature_names_)
            selected_features = check_correlation_threshold(
                data=data,
                feature_names=selected_features,
                features_gini_scores=self.features_gini_scores,
                corr_threshold=self.corr_threshold
            )

            self.model = selected_model.get_model(data[selected_features], target)

        # Copy final model attributes
        self.coef_ = selected_model.coef_
        self.intercept_ = selected_model.intercept_
        self.feature_names_ = selected_model.feature_names_
        self.model_score_ = selected_model.model_score_
        self.pvalues_ = selected_model.pvalues_

        return self.model

    def save_reports(self, path: str):
        save_reports(self.model, path)


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
