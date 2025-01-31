from typing import List, Union, Optional
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from joblib import Parallel, delayed

from .binning.functions import (cat_processing, find_cat_features,
                              num_processing, prepare_data, refit)
from .model.functions import (calc_model_results as _calc_model_results,
                            save_reports as _save_reports,
                            save_scorecard as _save_scorecard,
                            generate_sql as _generate_sql)
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
        self.woe_iv_dict: List = []
        self.feature_names: List[str] = []
        self.num_features: List[str] = []

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> 'WOETransformer':
        """Fit WOE transformer to data"""
        data, self.feature_names = prepare_data(data, self.special_cols)
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
            self.woe_iv_dict.extend(cat_results)
        else:
            self.num_features = self.feature_names

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

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using WOE encoding"""
        result = data.copy()
        transformed_features = []

        for woe_dict in self.woe_iv_dict:
            feature = next(iter(woe_dict))
            woe_info = woe_dict[feature]
            new_feature = f"{self.prefix}{feature}"
            transformed_features.append(new_feature)

            self._apply_woe_transform(result, feature, new_feature, woe_info, woe_dict)

            if not self.safe_original_data:
                del result[feature]

        return result[transformed_features] if not self.safe_original_data else result

    def _apply_woe_transform(self, data: pd.DataFrame, feature: str, new_feature: str,
                           woe_info: List, woe_dict: dict) -> None:
        """Apply WOE transformation to a single feature"""
        data[new_feature] = pd.NA
        for bin_info in woe_info:
            if feature in self.cat_features:
                data.loc[data[feature].isin(bin_info["bin"]), new_feature] = bin_info["woe"]
            else:
                mask = (data[feature] >= min(bin_info["bin"])) & (data[feature] < max(bin_info["bin"]))
                data.loc[mask, new_feature] = bin_info["woe"]

        self._handle_missing_values(data, new_feature, woe_info, woe_dict)

    def _handle_missing_values(self, data: pd.DataFrame, new_feature: str,
                             woe_info: List, woe_dict: dict) -> None:
        """Handle missing values in WOE transformation"""
        missing_bin = woe_dict.get("missing_bin")
        if missing_bin == "first":
            fill_value = woe_info[0]["woe"]
        elif missing_bin == "last":
            fill_value = woe_info[-1]["woe"]
        else:
            fill_value = woe_info[0]["woe"] if woe_info[0]["woe"] < woe_info[-1]["woe"] else woe_info[-1]["woe"]

        data[new_feature].fillna(fill_value, inplace=True)

    def refit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> None:
        """Refit WOE transformer with new data"""
        data, self.feature_names = prepare_data(data, self.special_cols)
        self.woe_iv_dict = Parallel(n_jobs=self.n_jobs)(
            delayed(refit)(
                data[list(d.keys())[0]], target,
                [b["bin"] for b in d[list(d.keys())[0]]],
                d["type_feature"],
                d["missing_bin"]
            ) for d in self.woe_iv_dict
        )

    def save_to_file(self, file_path: str) -> None:
        """Save WOE dictionary to file"""
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(self.woe_iv_dict, f, indent=4, cls=NpEncoder)

    def load_woe_iv_dict(self, file_path: str) -> None:
        """Load WOE dictionary from file"""
        with open(file_path, "r", encoding='utf-8') as f:
            self.woe_iv_dict = json.load(f)


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

        self.model_params = {k: v for k, v in locals().items() if k != 'self'}
        self.feature_selector: Optional[FeatureSelector] = None
        self.model: Optional[Model] = None
        self.model_results = None

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> 'CreateModel':
        """Fit model with feature selection"""
        self.feature_selector = FeatureSelector(**self.model_params)
        selected_features = self.feature_selector.select(data, target, self.model_params['feature_names'])

        self.model = Model(**self.model_params)
        self.model.get_model(data[selected_features], target)

        self.model_results = _calc_model_results(self.model)
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(data)

    def save_reports(self, path: str) -> None:
        """Save model reports"""
        if self.model is None:
            raise ValueError("Model must be fitted before saving reports")
        _save_reports(self.model, path)

    def generate_sql(self, encoder) -> str:
        """Generate SQL for model scoring"""
        if self.model is None:
            raise ValueError("Model must be fitted before generating SQL")
        return _generate_sql(encoder, self.model.feature_names_,
                           self.model.coef_, self.model.intercept_)

    def save_scorecard(self, encoder, path: str = '.',
                      base_scorecard_points: int = 444,
                      odds: int = 10,
                      points_to_double_odds: int = 69) -> None:
        """Save scorecard to file"""
        if self.model is None:
            raise ValueError("Model must be fitted before saving scorecard")
        _save_scorecard(self.model.feature_names_, encoder,
                       self.model_results, base_scorecard_points,
                       odds, points_to_double_odds, path)
