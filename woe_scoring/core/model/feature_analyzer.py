from typing import Dict, List, Union, Any # Any for now for encoder if not typed
from functools import partial # Keep for now, might change with direct function call
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Assuming gini_calculator.py is in the same directory or accessible via package structure
from . import gini_calculator # Relative import for sibling module

def calculate_gini_quality_for_features(data: Union[pd.DataFrame, np.ndarray],
                                        target: Union[pd.Series, np.ndarray],
                                        feature_names: List[str],
                                        random_state: int,
                                        class_weight: str,
                                        cv: int,
                                        scoring: str,
                                        n_jobs: int) -> Dict[str, float]:
    """Calculate Gini quality scores for multiple features in parallel."""

    # Ensure target is numpy array for consistency
    y_np: np.ndarray
    if isinstance(target, pd.Series):
        y_np = target.values
    elif isinstance(target, np.ndarray):
        y_np = target
    else:
        raise TypeError("Target must be pandas Series or numpy array.")

    # Data can be DataFrame or a dictionary of arrays if pre-processed.
    # For now, assume DataFrame as per original calculate_score usage.
    data_df: pd.DataFrame
    if isinstance(data, pd.DataFrame):
        data_df = data
    # If data is an ndarray, it implies it's a single feature, not suitable for this multi-feature function
    # Or it needs to be a structured array or dict of features. The original GiniScoreCalculator took DataFrame.
    # For now, sticking to DataFrame to match the expected input for gini_calculator.calculate_score's feature access.
    elif isinstance(data, np.ndarray):
         # This case needs clarification: if 'data' is a NumPy array, how are 'feature_names' used?
         # Original GiniScoreCalculator expects data[feature].
         # Assuming if data is ndarray, it's for a single pre-selected feature scenario,
         # which doesn't fit 'calc_features_gini_quality'.
         # For now, let's expect data to be a DataFrame here.
        raise TypeError("Input 'data' is expected to be a pandas DataFrame for multi-feature Gini calculation.")


    try:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Use the imported function directly
            # The 'partial' helps set arguments that are common across all calls
            calc_score_partial = partial(gini_calculator.calculate_score,
                                         data=data_df, # Pass DataFrame
                                         target=y_np,  # Pass numpy array target
                                         random_state=random_state,
                                         class_weight=class_weight,
                                         cv=cv,
                                         scoring=scoring,
                                         n_jobs=1) # Each call to calculate_score runs in its own thread

            # executor.map will raise the first exception encountered in a worker
            scores = list(executor.map(calc_score_partial, feature_names))
            return dict(zip(feature_names, scores))
    except RuntimeError as e: # Catch RuntimeError from gini_calculator.calculate_score
        # Log error: print(f"Error during Gini calculation for one or more features: {e}")
        # Re-raise with a more general message or handle by returning partial results if desired
        raise RuntimeError(f"Gini quality calculation failed: {e}") from e
    except Exception as e: # Catch other potential errors from ThreadPoolExecutor itself
        # Log error: print(f"Unexpected error in calculate_gini_quality_for_features: {e}")
        raise RuntimeError(f"Unexpected error during parallel Gini calculation: {e}") from e


def check_gini_threshold(feature_names: List[str],
                         gini_scores: Dict[str, float],
                         threshold: float) -> List[str]:
    """Get features that are above a specified Gini threshold."""
    return [f_name for f_name in feature_names if gini_scores.get(f_name, -np.inf) >= threshold]

def check_correlation(data: pd.DataFrame, # Expects DataFrame for .corr()
                      feature_names: List[str],
                      gini_scores: Dict[str, float],
                      threshold: float) -> List[str]:
    """Get a list of uncorrelated features, preferring features with higher Gini scores."""
    if not feature_names:
        return []

    try:
        # Ensure only valid feature names that exist in data are used for correlation
        valid_feature_names_for_corr = [name for name in feature_names if name in data.columns]
        if not valid_feature_names_for_corr:
            # Log: print("Warning: No valid features found in data for correlation check.")
            return []

        corr_matrix = data[valid_feature_names_for_corr].corr().abs()
        # Initialize with valid names that were actually used for correlation
        uncorrelated_features = set(valid_feature_names_for_corr)
    except (KeyError, TypeError, ValueError) as e:
        # Log error: print(f"Error calculating correlation matrix: {e}. Returning empty list.")
        # This could happen if feature_names are not in data, or data is non-numeric.
        raise ValueError(f"Failed to calculate correlation matrix. Ensure features exist and are numeric. Error: {e}") from e

    # Iterate over the upper triangle of the correlation matrix using valid_feature_names_for_corr
    for i in range(len(valid_feature_names_for_corr)):
        for j in range(i + 1, len(valid_feature_names_for_corr)):
            f1 = valid_feature_names_for_corr[i]
            f2 = valid_feature_names_for_corr[j]

            # No need to check if f1/f2 in uncorrelated_features if it's initialized with valid_feature_names_for_corr
            # and items are only removed.

            if corr_matrix.loc[f1, f2] >= threshold:
                gini1 = gini_scores.get(f1, -np.inf)
                gini2 = gini_scores.get(f2, -np.inf)

                if gini1 >= gini2:
                    uncorrelated_features.discard(f2)
                else:
                    uncorrelated_features.discard(f1)

    return list(uncorrelated_features)

def check_min_group_percentage(data: pd.DataFrame,
                               feature_names: List[str],
                               min_pct: float) -> List[str]:
    """Get features where all groups (categories/bins) meet a minimum percentage of the total population."""
    if min_pct < 0 or min_pct > 1:
        raise ValueError("min_pct must be between 0 and 1.")

    valid_features = []
    for f_name in feature_names:
        if f_name not in data.columns:
            # Log warning: print(f"Warning: Feature '{f_name}' not found in data for min_group_percentage check.")
            continue

        try:
            # Ensure column is not all NaN, as value_counts might behave unexpectedly or be empty.
            if data[f_name].isna().all():
                # Log: print(f"Info: Feature '{f_name}' is all NaN, skipping min_group_percentage check.")
                continue

            value_counts_norm = data[f_name].value_counts(normalize=True, dropna=False) # include NA in counts if needed

            if value_counts_norm.empty: # Should not happen if not all NaN, but as a safeguard
                # Log: print(f"Info: Feature '{f_name}' has no value counts, skipping.")
                continue

            # Check if the smallest group (excluding NaNs if dropna=True was used) meets threshold
            # If dropna=False, then NaN group percentage is also checked.
            # Decide if min_pct applies to NaN group. Assuming it does for now.
            if value_counts_norm.min() >= min_pct:
                valid_features.append(f_name)
        except Exception as e:
            # Log error: print(f"Error processing feature '{f_name}' for min_group_percentage: {e}")
            # Decide whether to skip this feature or raise error. For now, skip.
            continue

    return valid_features

def calculate_iv_for_feature(data: pd.DataFrame,
                             target: Union[pd.Series, np.ndarray],
                             feature_name: str) -> Dict[str, float]:
    """Calculate Information Value (IV) for a single categorical or binned feature."""

    y_np: np.ndarray
    if isinstance(target, pd.Series):
        y_np = target.values
    elif isinstance(target, np.ndarray):
        y_np = target
    else:
        raise TypeError("Target must be pandas Series or numpy array.")

    if feature_name not in data.columns:
        raise ValueError(f"Feature '{feature_name}' not found in DataFrame columns.")

    # Ensure data[feature_name] is not all NaN, which would make crosstab empty or problematic
    if data[feature_name].isna().all():
        # Log info: print(f"Info: Feature '{feature_name}' is all NaN. IV will be 0.")
        return {feature_name: 0.0}

    try:
        # Ensure y_np has more than one unique value for crosstab to be meaningful for binary target
        if len(np.unique(y_np)) <= 1:
            # Log warning: print(f"Warning: Target variable has only one unique value for feature '{feature_name}'. IV will be 0.")
            return {feature_name: 0.0}

        crosstab = pd.crosstab(data[feature_name], y_np, dropna=False) # dropna=False to include NaNs if present as a category
    except ValueError as e:
        # Log error: print(f"Error creating crosstab for feature '{feature_name}': {e}")
        raise ValueError(f"Failed to create crosstab for IV calculation on feature '{feature_name}'. Error: {e}") from e

    total_event = np.sum(y_np)
    total_nonevent = len(y_np) - total_event

    # Add 0.5 to avoid division by zero if a category has 0 events or non-events
    # This is a common adjustment for WOE/IV calculation.
    event_counts = crosstab.get(1, pd.Series(0, index=crosstab.index)) + 0.5
    nonevent_counts = crosstab.get(0, pd.Series(0, index=crosstab.index)) + 0.5

    # Ensure total_event and total_nonevent are also not zero for global rates
    safe_total_event = total_event if total_event > 0 else 0.5
    safe_total_nonevent = total_nonevent if total_nonevent > 0 else 0.5

    # Calculate WOE for each category/bin
    woe = np.log((event_counts / safe_total_event) / (nonevent_counts / safe_total_nonevent))

    # Calculate IV for each category/bin and sum up
    iv_per_bin = ((event_counts / safe_total_event) - (nonevent_counts / safe_total_nonevent)) * woe
    total_iv = iv_per_bin.sum()

    return {feature_name: total_iv}
