import os
from typing import List

import statsmodels.api as sm # For type hinting sm.Logit results

# Assuming Model class is defined elsewhere and will be imported where these functions are used.
# For example, if Model is in .model (sibling module to functions.py, now this module)
# from .model import Model # This line would be needed if Model type hint is strictly enforced at definition
# However, Python allows forward declaration / string type hints often.
# For now, using string literal 'Model' for type hint if Model class is complex or creates circular dependency.

def find_bad_model_features(
    feature_names: List[str],
    p_values: List[float],
    coefficients: List[float],
    p_value_threshold: float = 0.05,
    coeff_threshold: float = 0.0 # Coefficients for WOE transformed features should ideally be negative
    ) -> List[str]:
    """
    Find features with p-values above a threshold or coefficients above a threshold (e.g., positive for WOE).

    Args:
        feature_names: List of feature names.
        p_values: List of p-values corresponding to each feature.
        coefficients: List of coefficients corresponding to each feature.
        p_value_threshold: Threshold for p-values.
        coeff_threshold: Threshold for coefficients. For WOE, positive coeffs are usually bad.
                         Set to None or -infinity if no coefficient check is desired other than p-value.

    Returns:
        List of feature names considered "bad".
    """
    if not (len(feature_names) == len(p_values) == len(coefficients)):
        raise ValueError("Input lists (feature_names, p_values, coefficients) must have the same length.")

    bad_features = []
    for i, f_name in enumerate(feature_names):
        # Check p-value
        if p_values[i] > p_value_threshold:
            bad_features.append(f_name)
            continue # No need to check coefficient if p-value is already high

        # Check coefficient (only if p-value is acceptable)
        # For WOE transformed features, coefficients are typically expected to be negative.
        # A positive coefficient might indicate an issue with the WOE transformation or feature relationship.
        if coeff_threshold is not None and coefficients[i] > coeff_threshold:
            bad_features.append(f_name)

    return list(set(bad_features)) # Return unique list


def save_model_reports(model_summary_text: str, # Previously model.summary().as_text()
                       wald_test_summary_df: pd.DataFrame, # Previously model.wald_test_terms().summary_frame()
                       path: str = os.getcwd()) -> None:
    """
    Save model summary and Wald test reports to text files.

    Args:
        model_summary_text: The model summary as a string.
        wald_test_summary_df: The Wald test summary as a pandas DataFrame.
        path: Directory path to save reports. Defaults to current working directory.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except OSError as e:
        # Log error: print(f"Error creating directory {path}: {e}")
        raise IOError(f"Failed to create directory for reports at {path}: {e}") from e

    summary_path = os.path.join(path, "model_summary.txt")
    wald_path = os.path.join(path, "model_wald.txt")

    try:
        with open(summary_path, "w", encoding='utf-8') as f: # Added encoding
            f.write(model_summary_text)
    except IOError as e:
        # Log error: print(f"Error writing model summary to {summary_path}: {e}")
        raise IOError(f"Failed to write model summary report to {summary_path}: {e}") from e
    except Exception as e: # Catch any other unexpected error during write
        raise RuntimeError(f"An unexpected error occurred while saving summary report to {summary_path}: {e}") from e

    try:
        with open(wald_path, "w", encoding='utf-8') as f: # Added encoding
            wald_test_summary_df.to_string(f)
    except IOError as e:
        # Log error: print(f"Error writing Wald test summary to {wald_path}: {e}")
        raise IOError(f"Failed to write Wald test report to {wald_path}: {e}") from e
    except Exception as e: # Catch any other unexpected error (e.g. to_string issues)
        raise RuntimeError(f"An unexpected error occurred while saving Wald test report to {wald_path}: {e}") from e

# Note: calc_iv_dict was moved to feature_analyzer.py as calculate_iv_for_feature
# Note: The original find_bad_features took a Model object.
# The refactored version takes lists of feature_names, p_values, coefficients.
# This makes it more flexible as it doesn't depend on the specific Model class structure directly.
# The caller (likely in main.py or model.py) will be responsible for extracting these from its Model object.
# Similarly, save_model_reports now takes the direct string/DataFrame content.
import pandas as pd # Added missing import for pd.DataFrame in wald_test_summary_df type hint
