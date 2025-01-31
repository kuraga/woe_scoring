from typing import List, Union, Callable, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import l1_min_c
from functools import lru_cache


class SMWrapper(BaseEstimator, RegressorMixin):
    """A universal sklearn-style wrapper for statsmodels regressors"""

    def __init__(self) -> None:
        self.model_ = None

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> "SMWrapper":
        self.model_ = sm.Logit(target, sm.add_constant(data)).fit()
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        decision = self.model_.predict(sm.add_constant(data)) > 0.5
        return np.asarray(decision, dtype=np.int64)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        decision = self.model_.predict(sm.add_constant(data))
        return np.column_stack([1 - decision, decision])


class Model:
    def __init__(
            self,
            model_type: str,
            l1_exp_scale: int,
            l1_grid_size: int,
            cv: Optional[int] = None,
            class_weight: Optional[str] = None,
            random_state: Optional[int] = None,
            n_jobs: Optional[int] = None,
            scoring: Optional[str] = None
    ) -> None:
        self.model_type = model_type.lower()
        self.cv = cv
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.l1_exp_scale = l1_exp_scale
        self.l1_grid_size = l1_grid_size

        self.model = self._get_model(self.model_type)
        self.coef_: List[float] = []
        self.intercept_: float = 0.0
        self.feature_names_: List[str] = []
        self.model_score_: float = 0.0
        self.pvalues_: List[float] = []
        self.model_results: Any = None

    @lru_cache(maxsize=None)
    def get_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> Any:
        return self.model(data, target)

    def _get_model(self, model_type: str) -> Callable:
        model_types = {
            'sklearn': self._get_sklearn_model,
            'statsmodels': self._get_statsmodels_model
        }
        if model_type not in model_types:
            raise ValueError(f'Unknown model type: {model_type}. Should be either "sklearn" or "statsmodels"')
        return model_types[model_type]

    def _get_sklearn_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> Any:
        Cs = l1_min_c(data, target, loss="log", fit_intercept=True) * np.logspace(
            0, self.l1_exp_scale, self.l1_grid_size
        )
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=self.cv,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tol=1e-5,
            max_iter=5000,
            scoring=self.scoring
        ).fit(data, target)

        self.coef_ = list(model.coef_[0])
        self.intercept_ = model.intercept_[0]
        self.feature_names_ = list(data.columns)
        self.model_score_ = np.mean(cross_val_score(
            model, data, target, cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring
        ))
        self.pvalues_ = list(self._calc_pvalues(model, data))
        return model

    def _get_statsmodels_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> Any:
        model = SMWrapper().fit(data, target)
        if model.model_ is None:
            raise ValueError("Statsmodels model failed to fit")
        self.coef_ = list(model.model_.params[1:])
        self.intercept_ = model.model_.params[0]
        self.feature_names_ = list(data.columns)
        self.model_score_ = np.mean(cross_val_score(
            model, data, target, cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring
        ))
        self.pvalues_ = list(model.model_.pvalues)[1:]
        return model

    def _calc_pvalues(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        p = model.predict_proba(data)[:, 1]
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.insert(np.array(data), 0, 1, axis=1)
        ans = np.einsum('ij,ik,i->jk', x_full, x_full, p * (1 - p))
        vcov = np.linalg.inv(ans)
        se = np.sqrt(np.diag(vcov))
        t = coefs / se
        p = 2 * (1 - norm.cdf(np.abs(t)))
        return p[1:]
