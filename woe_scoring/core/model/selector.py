from operator import itemgetter
from functools import lru_cache
from typing import List, Union, Callable, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c

# Updated import for calculate_iv_for_feature
from .feature_analyzer import calculate_iv_for_feature


@dataclass
class FeatureSelector:
    """
    Initialize a feature selector object with the specified parameters.

    Args:
        selection_type (str): The type of feature selection algorithm to use.
        random_state (int): Random seed for reproducibility.
        class_weight (str): Class weights for imbalanced classification problems.
        cv (int): Number of cross-validation folds to use.
        n_jobs (int): Number of CPU cores to use for parallelization.
        max_vars (int): Maximum number of features to select.
        direction (str): The direction to select features in (forward or backward).
        scoring (str): The scoring metric to use for feature selection.
        l1_exp_scale (int): The exponent used for generating L1 regularization values.
        l1_grid_size (int): The size of the L1 regularization grid to search over.
        iv_threshold (float): The minimum information value threshold for a feature.
    """
    selection_type: str
    random_state: int
    class_weight: str
    cv: int
    n_jobs: int
    max_vars: int
    direction: str
    scoring: str
    l1_exp_scale: int
    l1_grid_size: int
    iv_threshold: float

    def __post_init__(self):
        self.selector = self._get_selector()
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input parameters"""
        if self.selection_type not in {'rfe', 'sfs', 'iv'}:
            raise ValueError(f'Unknown selection type: {self.selection_type}. Must be "rfe", "sfs" or "iv"')

    def select(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        if not feature_names:
            return []
        return self.selector(data=data, target=target, feature_names=feature_names)

    @lru_cache(maxsize=1)
    def _get_selector(self) -> Callable:
        """Returns the appropriate feature selection function based on selection_type."""
        selectors = {
            'rfe': self._select_by_rfe,
            'sfs': self._select_by_sfs,
            'iv': self._select_by_iv
        }
        return selectors[self.selection_type]

    def _get_base_estimator(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> LogisticRegression:
        """Creates and returns a LogisticRegression estimator with optimized parameters."""
        c_value = l1_min_c(data[feature_names], target, loss="log", fit_intercept=True)
        return LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tol=1e-5,
            max_iter=5000,
            penalty="l2",
            warm_start=True,
            C=c_value
        )

    def _select_by_iv(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """Selects top features based on Information Value (IV) score."""
        iv_dict: Dict[str, float] = {}
        for feature_name in feature_names:
            # Updated function call
            iv_dict.update(calculate_iv_for_feature(data, target, feature_name))

        sorted_features = sorted(iv_dict.items(), key=itemgetter(1), reverse=True)
        return [f for f, iv in sorted_features if iv >= self.iv_threshold][:self.max_vars]

    def _select_by_sfs(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """Selects the best features using Sequential Feature Selection."""
        selector = SequentialFeatureSelector(
            estimator=self._get_base_estimator(data, target, feature_names),
            n_features_to_select=min(self.max_vars, len(feature_names)),
            direction=self.direction,
            cv=self.cv,
            n_jobs=self.n_jobs,
            scoring=self.scoring
        )
        selector.fit(data[feature_names], target)
        return list(np.array(feature_names)[selector.get_support()])

    def _select_by_rfe(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray], feature_names: List[str]) -> List[str]:
        """Selects the best features using Recursive Feature Elimination with Cross-Validation."""
        selector = RFECV(
            estimator=self._get_base_estimator(data, target, feature_names),
            step=1,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=min(self.max_vars, len(feature_names)),
            n_jobs=self.n_jobs
        )
        selector.fit(data[feature_names], target)
        return list(np.array(feature_names)[selector.get_support()])
