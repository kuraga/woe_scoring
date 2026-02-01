import json
from itertools import combinations
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels

from .binning.functions import (cat_processing, find_cat_features,
                                num_processing, prepare_data, refit)
from .model.functions import (calc_features_gini_quality, check_correlation_threshold, check_features_gini_threshold,
                              check_min_pct_group, find_bad_features, generate_sql, save_reports, save_scorecard_fn)
from .model.model import Model
from .model.selector import FeatureSelector


class NpEncoder(json.JSONEncoder):
    """Convert NumPy objects to JSON serializable ones."""

    def default(self, obj):
        """Convert a non-serializable object to a serializable one.

        If `obj` is an instance of `np.integer`, this function returns it as a
        Python integer. If `obj` is an instance of `np.floating`, this function
        returns it as a Python float. If `obj` is an instance of `np.ndarray`,
        this function returns its contents as a nested list. Otherwise, this
        function delegates the conversion to the parent class.

        Args:
            obj: An object to be converted to a serializable one.

        Returns:
            A serializable version of `obj`.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Initializes the object for Weight of Evidence (WOE) encoding.

    Args:
        max_bins: Maximum number of bins allowed for each feature. Defaults to 10.
        min_pct_group: Minimum percentage of each bin allowed for each feature. Defaults to 0.05.
        n_jobs: Number of parallel jobs to run. Defaults to 1.
        prefix: Prefix to add to the name of each feature after encoding. Defaults to 'WOE_'.
        merge_type: Type of merging to use for continuous features. Possible values are 'chi2', 'woe' and 'monotonic'. Defaults to 'chi2'.
        cat_features: List of categorical features to encode. If not provided, all non-numeric features will be considered categorical. Defaults to None.
        special_cols: List of features to skip encoding. Defaults to None.
        cat_features_threshold: Maximum number of unique values allowed for a feature to be considered categorical. Defaults to 0.
        diff_woe_threshold: Minimum WOE difference allowed between any two adjacent bins for each feature. Defaults to 0.05.
        safe_original_data: Whether to keep a copy of the original data. Defaults to False.
    """

    def __init__(
            self,
            max_bins: Union[int, float] = 10,
            min_pct_group: float = 0.05,
            n_jobs: int = 1,
            prefix: str = "WOE_",
            merge_type: str = "chi2",
            cat_features: List[str] = None,
            special_cols: List[str] = None,
            cat_features_threshold: int = 0,
            diff_woe_threshold: float = 0.05,
            safe_original_data: bool = False,
            generate_features: bool = False,
            max_generated_features: int = 20,
    ):
        self.classes_ = None
        self.max_bins = max_bins
        self.min_pct_group = min_pct_group
        self.cat_features = cat_features
        self.special_cols = special_cols
        self.cat_features_threshold = cat_features_threshold
        self.diff_woe_threshold = diff_woe_threshold
        self.n_jobs = n_jobs
        self.prefix = prefix
        self.safe_original_data = safe_original_data
        self.merge_type = merge_type
        self.generate_features = generate_features
        self.max_generated_features = max_generated_features

        self.woe_iv_dict = []
        self.feature_names = []
        self.num_features = []
        self.generated_features_meta = []

    def _generate_features(
            self,
            data: pd.DataFrame,
            target: Union[pd.Series, np.ndarray],
            cat_features: List[str],
            num_features: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate new features from existing ones and select the best.
        """
        generated_features = {}
        generated_meta_candidates = []

        # 1. Ratio features
        for col_a, col_b in combinations(num_features, 2):
            feature_name = f"RATIO_{col_a}_DIV_{col_b}"
            # Avoid division by zero
            denom = data[col_b].replace(0, 1e-6)
            if pd.api.types.is_numeric_dtype(denom):
                generated_features[feature_name] = data[col_a] / denom
                generated_meta_candidates.append({
                    'name': feature_name,
                    'type': 'ratio',
                    'col_a': col_a,
                    'col_b': col_b
                })

        # 2. Categorical stats features
        for cat_col in cat_features:
            for num_col in num_features:
                feature_name = f"STATS_{num_col}_BY_{cat_col}_MEAN"
                # Calculate mean
                generated_features[feature_name] = data.groupby(cat_col)[num_col].transform('mean')
                # Store mapping for transform
                mapping = data.groupby(cat_col)[num_col].mean().to_dict()
                generated_meta_candidates.append({
                    'name': feature_name,
                    'type': 'stats',
                    'cat_col': cat_col,
                    'num_col': num_col,
                    'func': 'mean',
                    'mapping': mapping
                })

        if not generated_features:
            return data, []

        # Create DataFrame from generated features
        gen_df = pd.DataFrame(generated_features, index=data.index)

        # Calculate Gini for selection
        gini_scores = calc_features_gini_quality(
            data=gen_df,
            target=target,
            feature_names=list(gen_df.columns),
            random_state=42,
            class_weight='balanced',
            cv=3,
            scoring='roc_auc',
            n_jobs=self.n_jobs
        )

        # Select best features
        sorted_features = sorted(gini_scores.items(), key=lambda x: x[1], reverse=True)
        best_features = [f[0] for f in sorted_features[:self.max_generated_features]]

        # Filter metadata
        self.generated_features_meta = [
            meta for meta in generated_meta_candidates
            if meta['name'] in best_features
        ]

        # Add best features to data
        data = pd.concat([data, gen_df[best_features]], axis=1)

        return data, best_features

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> None:
        """
        Fits the WoE transformer to the provided data and target variables.

        :param data: A pandas DataFrame with the input data.
        :param target: A pandas Series or numpy array with the target variable.
        :return: None
        """
        special_cols = self.special_cols or []
        cat_features = self.cat_features or []

        data, self.feature_names = prepare_data(data=data, special_cols=special_cols)
        self.classes_ = unique_labels(target)

        if len(cat_features) == 0:
            cat_features = find_cat_features(
                x=data,
                feature_names=self.feature_names,
                cat_features_threshold=self.cat_features_threshold
            )

        if self.generate_features:
            # Identify numeric features for generation
            num_features_for_gen = [
                feature for feature in self.feature_names
                if feature not in cat_features
            ]
            
            data, generated_features_list = self._generate_features(
                data=data,
                target=target,
                cat_features=cat_features,
                num_features=num_features_for_gen
            )
            # Update feature names with generated ones
            self.feature_names.extend(generated_features_list)
            # Generated features are numeric, so cat_features remains same
        
        self.cat_features_ = cat_features

        if len(cat_features) > 0:
            self.num_features = [
                feature
                for feature in self.feature_names
                if feature not in cat_features
            ]
            self.woe_iv_dict = Parallel(n_jobs=self.n_jobs)(
                delayed(cat_processing)(
                    data[col],
                    target,
                    self.min_pct_group,
                    self.max_bins,
                    self.diff_woe_threshold
                ) for col in cat_features
            )
        else:
            self.num_features = self.feature_names

        num_features_res = Parallel(n_jobs=self.n_jobs)(
            delayed(num_processing)(
                data[col],
                target,
                self.min_pct_group,
                self.max_bins,
                self.diff_woe_threshold,
                self.merge_type
            ) for col in self.num_features
        )

        self.woe_iv_dict += num_features_res
        
        # Store effective cat_features for transform
        self.cat_features_ = cat_features
        
        # Append generated features metadata and categorical features list to woe_iv_dict for persistence
        self.woe_iv_dict.append({
            'metadata': {
                'generated_features_meta': self.generated_features_meta,
                'cat_features': self.cat_features_
            }
        })

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
        
        # Check for metadata
        generated_meta = []
        clean_woe_iv_dict = []
        cat_features_from_meta = None
        
        for item in self.woe_iv_dict:
            if 'metadata' in item:
                generated_meta = item['metadata'].get('generated_features_meta', [])
                cat_features_from_meta = item['metadata'].get('cat_features')
            elif 'generated_features_meta' in item: # Backward compatibility for previous step
                generated_meta = item['generated_features_meta']
            else:
                clean_woe_iv_dict.append(item)
        
        if generated_meta:
             self.generated_features_meta = generated_meta
        
        # Use fitted cat_features if available
        if cat_features_from_meta is not None:
            self.cat_features_ = cat_features_from_meta
        
        cat_features = getattr(self, 'cat_features_', self.cat_features or [])

        # Generate features if metadata exists
        if self.generated_features_meta:
            for meta in self.generated_features_meta:
                if meta['type'] == 'ratio':
                    col_a = meta['col_a']
                    col_b = meta['col_b']
                    if col_a in data.columns and col_b in data.columns:
                        denom = data[col_b].replace(0, 1e-6)
                        data[meta['name']] = data[col_a] / denom
                elif meta['type'] == 'stats':
                    cat_col = meta['cat_col']
                    mapping = meta['mapping']
                    if cat_col in data.columns:
                        # Apply mapping
                        data[meta['name']] = data[cat_col].map(mapping)

        # Use fitted cat_features if available
        cat_features = getattr(self, 'cat_features_', self.cat_features or [])

        # Pre-create all new feature columns
        for woe_iv in clean_woe_iv_dict:
            feature = list(woe_iv)[0]
            new_feature = self.prefix + feature
            data[new_feature] = np.nan
            if not self.safe_original_data:
                features_to_delete.append(feature)

        # Apply transformations
        for woe_iv in clean_woe_iv_dict:
            feature = list(woe_iv)[0]
            woe_iv_feature = woe_iv[feature]
            new_feature = self.prefix + feature
            
            if not woe_iv_feature:
                continue

            # Apply bins based on feature type
            if feature in cat_features:
                # Categorical features - vectorized approach using map with default to NaN
                bin_map = {}
                for bin_values in woe_iv_feature:
                    for bin_val in bin_values["bin"]:
                        bin_map[bin_val] = bin_values["woe"]

                # Convert to category first for efficiency with large datasets
                data.loc[:, new_feature] = data[feature].map(bin_map)
            else:
                # Numerical features - Optimized using pd.cut
                if len(woe_iv_feature) > 0:
                    # Construct bin edges from contiguous intervals
                    # bin_values["bin"] is [min, max]
                    edges = [woe_iv_feature[0]["bin"][0]] + [b["bin"][1] for b in woe_iv_feature]
                    labels = [b["woe"] for b in woe_iv_feature]
                    
                    # pd.cut handles binning efficiently
                    # right=False means intervals are [a, b) matching original logic
                    # include_lowest=True extends the first interval to be [a, b) as well (mostly relevant if data == min)
                    data[new_feature] = pd.cut(
                        data[feature],
                        bins=edges,
                        labels=labels,
                        right=False,
                        include_lowest=True,
                        ordered=False
                    )
                    
                    # Convert categorical result to float (pd.cut returns category/object)
                    # Use to_numeric or astype, but handle NaNs gracefully (they remain NaN)
                    if data[new_feature].dtype.name == 'category':
                        data[new_feature] = data[new_feature].astype(float)
                    else:
                        data[new_feature] = pd.to_numeric(data[new_feature], errors='coerce')
                else:
                    # No bins, everything is NaN (already initialized)
                    pass

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

        Returns:
            None
        """
        with open(file_path, "w") as f:
            json.dump(self.woe_iv_dict, f, indent=4, cls=NpEncoder)

    def load_woe_iv_dict(self, file_path: str) -> None:
        """
        Load a dictionary of WoE and IV values from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            None
        """
        with open(file_path, "r") as json_file:
            self.woe_iv_dict = json.load(json_file)

    def refit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> None:
        """
        Refit the model using new data and target.

        Args:
            data (pd.DataFrame): The input data to be prepared and used for refitting the model.
            target (Union[pd.Series, np.ndarray]): The target variable used for fitting the model.

        Returns:
            None
        """

        data, self.feature_names = prepare_data(data=data, special_cols=self.special_cols)

        # Ensure target is numpy array for consistency
        target_values = target.values if hasattr(target, 'values') else np.array(target)
        
        # Preserve metadata
        metadata = None
        for item in self.woe_iv_dict:
            if 'metadata' in item:
                metadata = item
                break

        # Process in parallel with optimized parameters
        self.woe_iv_dict = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(refit)(
                data[list(woe_iv.keys())[0]],
                target_values,
                [_bin["bin"] for _bin in woe_iv[list(woe_iv.keys())[0]]],
                woe_iv["type_feature"],
                woe_iv["missing_bin"]
            ) for woe_iv in self.woe_iv_dict if 'metadata' not in woe_iv
        )
        
        # Restore metadata
        if metadata:
            self.woe_iv_dict.append(metadata)


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

    def __init__(
            self,
            selection_method: str = 'rfe',
            model_type: str = 'sklearn',
            max_vars: Union[int, float, None] = None,
            special_cols: List[str] = None,
            unused_cols: List[str] = None,
            n_jobs: int = 1,
            gini_threshold: float = 5.0,
            iv_threshold: float = 0.05,
            corr_threshold: float = 0.5,
            min_pct_group: float = 0.05,
            random_state: int = None,
            class_weight: str = 'balanced',
            direction: str = "forward",
            cv: int = 3,
            l1_exp_scale: int = 4,
            l1_grid_size: int = 20,
            scoring: str = "roc_auc",
            vif_threshold: float = 10.0,
    ):
        self.selection_method = selection_method
        self.model_type = model_type
        self.max_vars = max_vars
        self.special_cols = special_cols
        self.unused_cols = unused_cols
        self.n_jobs = n_jobs
        self.gini_threshold = gini_threshold
        self.iv_threshold = iv_threshold
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.min_pct_group = min_pct_group
        self.random_state = random_state
        self.class_weight = class_weight
        self.direction = direction
        self.cv = cv
        self.l1_exp_scale = l1_exp_scale
        self.l1_grid_size = l1_grid_size
        self.scoring = scoring

        self.features_gini_scores = {}

        self.coef_ = []
        self.intercept_ = None
        self.feature_names_ = []
        self.model_score_ = None
        self.pvalues_ = []

        self.model = None

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> None:
        """
        Fit the model with the given data and target.

        Args:
            data (pd.DataFrame): The input data.
            target (Union[pd.Series, np.ndarray]): The target values.

        Returns:
            The fitted model.
        """
        # Resolve parameters
        special_cols = self.special_cols or []
        unused_cols = self.unused_cols or []

        # Prepare data and filter features
        data, self.feature_names_ = prepare_data(data=data, special_cols=special_cols)

        # Remove unused columns if specified
        if unused_cols:
            self.feature_names_ = [f for f in self.feature_names_ if f not in unused_cols]

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
            iv_threshold=self.iv_threshold,
            vif_threshold=self.vif_threshold
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

        # Construct model_results DataFrame for save_scorecard
        # Expected structure:
        # Column 0: Feature names (starting with 'const')
        # Column 'coef': Coefficients
        # Column 'P>|z|': P-values
        
        # Create lists for DataFrame construction
        res_features = ['const'] + self.feature_names_
        res_coefs = [self.intercept_] + self.coef_
        # Pad p-values with NaN for intercept if not available, or 0.0
        res_pvalues = [0.0] + self.pvalues_
        
        self.model_results = pd.DataFrame({
            'const': res_features, # Column name doesn't matter for iloc[:, 0], but naming it 'const' or 'Feature' is cleaner
            'coef': res_coefs,
            'P>|z|': res_pvalues
        })

        return self

    def save_reports(self, path: str):
        save_reports(self.model, path)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict(data)

    def generate_sql(self, encoder) -> str:
        return generate_sql(
            encoder=encoder,
            feature_names=self.feature_names_,
            coef=self.coef_,
            intercept=self.intercept_
        )

    def save_scorecard(
            self,
            encoder,
            path: str = '.',
            base_scorecard_points: int = 444,
            odds: int = 10,
            points_to_double_odds: int = 69,
    ):
        save_scorecard_fn(
            feature_names=self.feature_names_,
            encoder=encoder,
            model_results=self.model_results,
            base_scorecard_points=base_scorecard_points,
            odds=odds,
            points_to_double_odds=points_to_double_odds,
            path=path
        )
