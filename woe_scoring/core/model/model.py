from typing import List, Union, Callable, Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import l1_min_c
# from functools import lru_cache


class SMWrapper(BaseEstimator, RegressorMixin):
    """A universal sklearn-style wrapper for statsmodels regressors"""

    def __init__(self) -> None:
        self.model_ = None

    def fit(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> "SMWrapper":
        try:
            # sm.add_constant can raise error if dataframe is empty or has problematic dtypes
            X_const = sm.add_constant(data)
            self.model_ = sm.Logit(target, X_const).fit()
        except Exception as e:
            # Log error: print(f"Error fitting statsmodels Logit: {e}")
            raise RuntimeError(f"Failed to fit statsmodels Logit model: {e}") from e
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        try:
            X_const = sm.add_constant(data)
            decision_probs = self.model_.predict(X_const)
            decision = decision_probs > 0.5
            return np.asarray(decision, dtype=np.int64)
        except Exception as e:
            # Log error: print(f"Error in SMWrapper predict: {e}")
            raise RuntimeError(f"Statsmodels prediction failed: {e}") from e

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        try:
            X_const = sm.add_constant(data)
            decision_probs = self.model_.predict(X_const)
            return np.column_stack([1 - decision_probs, decision_probs])
        except Exception as e:
            # Log error: print(f"Error in SMWrapper predict_proba: {e}")
            raise RuntimeError(f"Statsmodels probability prediction failed: {e}") from e


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

    # @lru_cache(maxsize=None)
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
        try:
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

            # cross_val_score can also fail
            cv_scores = cross_val_score(
                model, data, target, cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring
            )
            self.model_score_ = np.nanmean(cv_scores) if cv_scores is not None and len(cv_scores) > 0 else 0.0

            self.pvalues_ = list(self._calc_pvalues(model, data))
            return model
        except Exception as e:
            # Log error: print(f"Error fitting sklearn LogisticRegressionCV model: {e}")
            raise RuntimeError(f"Failed to fit sklearn model: {e}") from e

    def _get_statsmodels_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> Any:
        try:
            model_wrapper = SMWrapper().fit(data, target) # SMWrapper.fit now handles its own errors
            if model_wrapper.model_ is None: # Should be caught by SMWrapper.fit, but double check
                raise ValueError("Statsmodels model (SMWrapper.model_) is None after fitting.")

            actual_model = model_wrapper.model_
            self.coef_ = list(actual_model.params[1:])
            self.intercept_ = actual_model.params[0]
            self.feature_names_ = list(data.columns)

            # cross_val_score can also fail
            cv_scores = cross_val_score(
                model_wrapper, data, target, cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring
            ) # Pass the wrapper
            self.model_score_ = np.nanmean(cv_scores) if cv_scores is not None and len(cv_scores) > 0 else 0.0

            self.pvalues_ = list(actual_model.pvalues)[1:]
            return model_wrapper # Return the wrapper consistent with sklearn model return
        except Exception as e:
            # Log error: print(f"Error fitting statsmodels model via SMWrapper: {e}")
            raise RuntimeError(f"Failed to fit statsmodels model: {e}") from e
        return model

    def _calc_pvalues(self, model: Any, data: pd.DataFrame) -> np.ndarray:
        try:
            # Ensure model has predict_proba, intercept_, and coef_ attributes
            if not all(hasattr(model, attr) for attr in ['predict_proba', 'intercept_', 'coef_']):
                # Log: print("Warning: Model for p-value calculation lacks required attributes. Returning empty p-values.")
                return np.array([]) # Or array of NaNs matching number of coeffs

            pred_probs = model.predict_proba(data)[:, 1]

            # Ensure intercept_ and coef_ are available and are numbers/arrays of numbers
            intercept = model.intercept_
            coeffs_array = model.coef_[0] # Assuming coef_ is 2D for LogisticRegressionCV

            if not (isinstance(intercept, (int, float, np.number)) and isinstance(coeffs_array, np.ndarray)):
                 # Log: print("Warning: Model intercept_ or coef_ have unexpected types. Returning empty p-values.")
                return np.array([])

            model_coefficients = np.concatenate([[intercept], coeffs_array])

            # Ensure data can be converted to numeric array for x_full
            x_data_array = np.array(data, dtype=float)
            x_full = np.insert(x_data_array, 0, 1, axis=1) # Add column for intercept

            # Fisher Information Matrix calculation
            # p * (1-p) can be zero if probabilities are 0 or 1. Add small epsilon for stability if needed.
            p_times_1_minus_p = pred_probs * (1 - pred_probs)
            # Add a small epsilon to avoid exact zeros in variance calculation if necessary,
            # though typically sklearn/statsmodels handles this.
            # p_times_1_minus_p = np.maximum(p_times_1_minus_p, 1e-9)

            if x_full.shape[0] != len(p_times_1_minus_p):
                raise ValueError("Shape mismatch between x_full and probability weights for p-value calculation.")
            if x_full.shape[1] != len(model_coefficients):
                 raise ValueError("Shape mismatch between x_full columns and number of coefficients.")


            fisher_info_matrix_comp = np.einsum('ij,ik,i->jk', x_full, x_full, p_times_1_minus_p)

            if np.linalg.det(fisher_info_matrix_comp) == 0:
                # Log: print("Warning: Fisher Information Matrix is singular. Cannot calculate p-values via matrix inversion.")
                return np.full(len(coeffs_array), np.nan) # Return NaNs for p-values

            variance_covariance_matrix = np.linalg.inv(fisher_info_matrix_comp)
            standard_errors = np.sqrt(np.diag(variance_covariance_matrix))

            z_scores = model_coefficients / standard_errors
            p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

            return p_values[1:] # Exclude p-value for intercept
        except np.linalg.LinAlgError as e:
            # Log error: print(f"LinAlgError in p-value calculation (e.g., singular matrix): {e}")
            # Return NaNs for p-values if matrix inversion fails
            num_coeffs = len(model.coef_[0]) if hasattr(model, 'coef_') and model.coef_ is not None and len(model.coef_) > 0 else 0
            return np.full(num_coeffs, np.nan)
        except Exception as e:
            # Log error: print(f"Unexpected error in _calc_pvalues: {e}")
            # Fallback to returning NaNs or empty array for p-values
            num_coeffs = len(model.coef_[0]) if hasattr(model, 'coef_') and model.coef_ is not None and len(model.coef_) > 0 else 0
            return np.full(num_coeffs, np.nan) # Return NaNs matching number of coefficients
