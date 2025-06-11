from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# functools.lru_cache can be added if this function is called with same heavy computation repeatedly
# from functools import lru_cache

# @lru_cache(maxsize=128) # Add if performance profiling suggests it's beneficial
def calculate_score(data: Union[pd.DataFrame, np.ndarray],
                    target: Union[pd.Series, np.ndarray],
                    feature: str,
                    random_state: int,
                    class_weight: str,
                    cv: int,
                    scoring: str,
                    n_jobs: int) -> float:
    """Calculate the Gini score for a given feature using Logistic Regression."""

    model = LogisticRegression(random_state=random_state,
                             class_weight=class_weight,
                             max_iter=1000,
                             n_jobs=n_jobs, # n_jobs in LogisticRegression is for solver if applicable
                             warm_start=True)

    # Ensure X is 2D for scikit-learn
    if isinstance(data, pd.DataFrame):
        X = data[feature].values.reshape(-1, 1)
    elif isinstance(data, np.ndarray): # Assuming data is already the feature column
        X = data.reshape(-1, 1)
    else:
        raise TypeError("Data must be pandas DataFrame or numpy array.")

    # Ensure target is 1D array
    if isinstance(target, pd.Series):
        y = target.values
    elif isinstance(target, np.ndarray):
        y = target
    else:
        raise TypeError("Target must be pandas Series or numpy array.")

    # n_jobs in cross_val_score is for parallelizing CV folds
    try:
        scores = cross_val_score(estimator=model,
                               X=X,
                               y=y,
                               cv=cv,
                               scoring=scoring,
                               n_jobs=n_jobs) # This n_jobs is for cross_val_score
    except ValueError as e:
        # Example: CV might fail if a class is not present in a fold
        # Log error: print(f"Error during cross_val_score for feature '{feature}': {e}")
        # Return a very low Gini or raise a custom error
        # For now, returning a low Gini score (e.g., -100 or 0, Gini is usually positive)
        # A Gini of 0 means no predictive power, -100 indicates failure.
        # Let's return 0, implying no separation, or re-raise.
        # Re-raising is often better for library code so user knows.
        raise RuntimeError(f"Cross-validation failed for feature '{feature}'. Check data for this feature, target balance, or CV folds. Original error: {e}") from e
    except Exception as e: # Catch any other unexpected errors
        # Log error: print(f"Unexpected error during cross_val_score for feature '{feature}': {e}")
        raise RuntimeError(f"Unexpected error in Gini calculation for feature '{feature}': {e}") from e


    # Check if scores array is empty or contains NaNs, which can happen if scoring fails in all folds
    if scores is None or len(scores) == 0 or np.isnan(scores).all():
        # Log warning: print(f"Warning: Gini calculation for feature '{feature}' resulted in no valid scores.")
        return 0.0 # Or a more distinct error indicator like -100.0

    mean_score = np.nanmean(scores) # Use nanmean to handle potential NaNs from some folds failing
    return (mean_score * 2 - 1) * 100
