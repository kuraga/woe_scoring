from typing import Dict, List, Any # 'Any' for encoder type
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np # For np.log

# This function was originally a top-level _calc_score_points
def calculate_scorecard_points(woe: float,
                               coef: float,
                               intercept: float,
                               factor: float,
                               offset: float,
                               n_features: int) -> float:
    """Calculate scorecard points for a given WOE value and model parameters."""
    # Ensure intercept is divided by n_features only once per feature's set of bins,
    # or handle globally if intercept_per_feature is part of overall score calculation logic.
    # The original formula: -(woe * coef + intercept / n_features) * factor + offset / n_features
    # This implies a portion of intercept and offset is allocated to each feature's characteristic points.
    if n_features <= 0:
        # Log warning or raise error: print(f"Warning: n_features is {n_features}, which is invalid for point calculation. Returning 0 points.")
        # This situation might indicate an issue upstream (e.g. model with no features).
        # Returning a neutral value like 0 or NaN, or raising error.
        # For now, let's make it somewhat neutral by ignoring the per-feature part of offset/intercept.
        return -(woe * coef) * factor # Simplified, but might not be ideal.
                                     # A ValueError might be more appropriate.
        # raise ValueError("n_features must be positive for scorecard point calculation.")

    return -(woe * coef + intercept / n_features) * factor + (offset / n_features)


def _update_base_scorecard_stats(stats_dict: Dict[str, List],
                                 feature_name: str,
                                 model_results_df: pd.DataFrame,
                                 feature_row_idx_in_model_results: int
                                 ) -> None:
    """Update basic feature statistics in the stats_dict."""
    try:
        stats_dict["feature"].append(feature_name.replace("WOE_", ""))
        stats_dict["coef"].append(model_results_df.loc[feature_row_idx_in_model_results, "coef"])
        stats_dict["pvalue"].append(model_results_df.loc[feature_row_idx_in_model_results, "P>|z|"])
    except KeyError as e:
        # Log error: print(f"KeyError accessing model_results_df for feature {feature_name}: {e}. Check column names ('coef', 'P>|z|').")
        # Fill with NaN or re-raise, depending on desired strictness.
        # For now, let it propagate if columns are fundamentally missing.
        raise KeyError(f"Missing expected columns ('coef' or 'P>|z|') in model_results_df. Error: {e}") from e
    except IndexError as e:
        # Log error: print(f"IndexError accessing model_results_df for feature {feature_name} at index {feature_row_idx_in_model_results}: {e}.")
        raise IndexError(f"Invalid index for model_results_df for feature {feature_name}. Error: {e}") from e


def _update_detailed_feature_scorecard_stats(stats_dict: Dict[str, List],
                                             feature_woe_rules: List[Dict], # List of bin dicts for the feature
                                             feature_coef: float,
                                             model_intercept: float,
                                             factor: float,
                                             offset: float,
                                             n_model_features: int) -> None:
    """Update detailed WOE, IV, and scorecard points for each bin of a feature."""
    for bin_info in feature_woe_rules: # bin_info is a dict per bin
        bin_values = bin_info.get("bin", []) # Default to empty list if 'bin' key missing

        # Represent bin values as a string, similar to original logic
        # Special handling for -1 (often a missing placeholder in original context)
        # This part might need alignment with how binning module outputs bin representations,
        # especially for "Missing" string placeholders if they exist.
        bin_values_str_list = []
        if isinstance(bin_values, list):
            for v in bin_values:
                if v == -1 or (isinstance(v, str) and v == "Missing"): # Match original replacement logic
                    bin_values_str_list.append("missing")
                elif isinstance(v, (np.floating, float)) and np.isinf(v):
                    bin_values_str_list.append(str(v)) # 'inf', '-inf'
                else:
                    bin_values_str_list.append(str(v))
        else: # If bin_values is not a list (e.g. single float for numerical upper bound, though less likely here)
            bin_values_str_list.append(str(bin_values))

        stats_dict["bin"].append(", ".join(bin_values_str_list) if bin_values_str_list else str(bin_values))


        stats_dict["WOE"].append(bin_info.get("woe"))
        stats_dict["IV"].append(bin_info.get("iv"))
        stats_dict["percent_of_population"].append(bin_info.get("pct"))
        stats_dict["total"].append(bin_info.get("total"))
        stats_dict["event_cnt"].append(bin_info.get("bad"))

        non_event_count = bin_info.get("total", 0) - bin_info.get("bad", 0)
        stats_dict["non_event_cnt"].append(non_event_count)
        stats_dict["event_rate"].append(bin_info.get("bad_rate"))

        points = calculate_scorecard_points(
            woe=bin_info.get("woe", 0.0), # Default WOE to 0 if missing
            coef=feature_coef,
            intercept=model_intercept,
            factor=factor,
            offset=offset,
            n_features=n_model_features
        )
        stats_dict["score_ball"].append(points)

def _calculate_feature_scorecard_stats(
    feature_idx_in_model_results: int, # Index used for accessing row in model_results_df
    woe_feature_name_from_model: str, # Name from model_results (e.g. "WOE_Age" or "const")
    all_woe_feature_names_in_model: List[str], # All WOE features in the model, e.g. ["WOE_Age", "WOE_Income"]
    woe_encoder_rules: Any, # The encoder object or dict containing all WOE rules (e.g. encoder.woe_iv_dict)
    model_results_df: pd.DataFrame, # DataFrame of model coeffs, pvalues (const included)
    factor: float,
    offset: float
    ) -> pd.DataFrame:
    """
    Calculate scorecard statistics for a single feature (or 'const').
    """
    stats_result_dict = {
        "feature": [], "coef": [], "pvalue": [], "bin": [], "WOE": [], "IV": [],
        "percent_of_population": [], "total": [], "event_cnt": [],
        "non_event_cnt": [], "event_rate": [], "score_ball": [],
    }

    # Update base stats like feature name, coefficient, p-value
    _update_base_scorecard_stats(stats_result_dict, woe_feature_name_from_model, model_results_df, feature_idx_in_model_results)

    # Ensure 'const' row exists and 'coef' column exists before trying to access intercept
    const_row = model_results_df[model_results_df.iloc[:, 0] == 'const']
    if const_row.empty:
        raise ValueError("Could not find 'const' row in model_results_df to extract intercept.")
    if "coef" not in const_row.columns:
        raise ValueError("'coef' column missing in model_results_df for 'const' row.")
    try:
        model_intercept = const_row["coef"].iloc[0]
    except IndexError: # Should be caught by const_row.empty, but for safety
        raise ValueError("Failed to extract intercept from 'const' row in model_results_df.")

    num_model_features = len(all_woe_feature_names_in_model)
    if num_model_features == 0 and woe_feature_name_from_model != 'const':
        # Log warning: print("Warning: num_model_features is 0. Scorecard points might be misleading.")
        # calculate_scorecard_points will handle n_features=0 if it's not caught before.
        # Allow calculation to proceed but be aware.
        pass


    if woe_feature_name_from_model == 'const': # Intercept row
        # For 'const', fill other fields with placeholders like "-"
        for key in stats_result_dict:
            if key not in ["feature", "coef", "pvalue"]:
                # Pad list if it's shorter (can happen if _update_base_stats added first items)
                if not stats_result_dict[key]:
                     stats_result_dict[key].append("-")
    else:
        # This is an actual feature, find its WOE rules and calculate detailed stats
        original_feature_name = woe_feature_name_from_model.replace("WOE_", "")

        feature_specific_woe_rules = None
        # woe_encoder_rules is expected to be like original encoder.woe_iv_dict (list of dicts)
        if not isinstance(woe_encoder_rules, list):
            # Log error: print(f"Error: woe_encoder_rules is not a list for feature {original_feature_name}.")
            # Fill with placeholders as rules are missing/malformed.
            feature_specific_woe_rules = None
        else:
            for feature_rule_set in woe_encoder_rules:
                if not isinstance(feature_rule_set, dict):
                    # Log warning: print(f"Warning: Item in woe_encoder_rules is not a dict for {original_feature_name}. Skipping.")
                    continue
                if original_feature_name in feature_rule_set:
                    feature_specific_woe_rules = feature_rule_set[original_feature_name]
                    if not isinstance(feature_specific_woe_rules, list):
                        # Log warning: print(f"Warning: Rules for {original_feature_name} are not a list. Skipping.")
                        feature_specific_woe_rules = None # Invalid format
                    break

        if feature_specific_woe_rules:
            current_feature_coefficient = stats_result_dict["coef"][-1] # Coeff already added by _update_base_stats

            # Ensure feature_specific_woe_rules is a list of dicts
            if not all(isinstance(item, dict) for item in feature_specific_woe_rules):
                # Log error: print(f"Error: Not all items in feature_specific_woe_rules for {original_feature_name} are dictionaries.")
                # Fill with placeholders
                for key in stats_result_dict:
                    if key not in ["feature", "coef", "pvalue"] and not stats_result_dict[key]:
                        stats_result_dict[key].append("-")
                return pd.DataFrame.from_dict(stats_result_dict)

            _update_detailed_feature_scorecard_stats(
                stats_result_dict,
                feature_specific_woe_rules,
                current_feature_coefficient,
                model_intercept,
                factor,
                offset,
                num_model_features
            )
        else:
            # No WOE rules found for this feature, fill with placeholders
            for key in stats_result_dict:
                if key not in ["feature", "coef", "pvalue"] and not stats_result_dict[key]:
                    stats_result_dict[key].append("-") # Or specific error marker

    return pd.DataFrame.from_dict(stats_result_dict)


def build_scorecard_data(
    woe_feature_names_in_model: List[str], # Actual features in model, e.g. ["WOE_Age", "WOE_Income"]
    woe_encoder_rules: Any, # Contains all WOE rules (e.g., encoder.woe_iv_dict)
    model_results_df: pd.DataFrame, # DataFrame from calc_model_results (includes 'const')
    factor: float,
    offset: float
    ) -> List[pd.DataFrame]:
    """
    Builds a list of DataFrames, each containing scorecard statistics for a feature or 'const'.

    Args:
        woe_feature_names_in_model: List of WOE-transformed feature names used in the model.
        woe_encoder_rules: The collection of WOE transformation rules for all features.
        model_results_df: DataFrame containing model coefficients and p-values, including a row for 'const'.
                           Expected columns in model_results_df: first column for names (e.g. 'const', 'WOE_Age'), 'coef', 'P>|z|'.
        factor: Scorecard scaling factor.
        offset: Scorecard scaling offset.

    Returns:
        A list of pandas DataFrames, where each DataFrame details one feature's scorecard contribution.
    """
    scorecard_dfs = []

    # model_results_df.iloc[:, 0] contains feature names like 'const', 'WOE_Feature1', ...
    # Iterate through each row of model_results_df (which includes 'const' and all WOE features)
    try:
        with ThreadPoolExecutor() as executor:
            futures = []
            # Ensure model_results_df has at least one column before iloc[:, 0]
            if model_results_df.empty or model_results_df.shape[1] == 0:
                raise ValueError("model_results_df is empty or has no columns.")

            for idx, feature_name_in_row in enumerate(model_results_df.iloc[:, 0]):
                future = executor.submit(
                    _calculate_feature_scorecard_stats,
                    idx,
                    feature_name_in_row,
                    woe_feature_names_in_model,
                    woe_encoder_rules,
                    model_results_df,
                    factor,
                    offset
                )
                futures.append(future)

            for future in futures:
                scorecard_dfs.append(future.result()) # Exceptions from workers are raised here

        return scorecard_dfs
    except Exception as e:
        # Log error: print(f"Error building scorecard data: {e}")
        # Re-raise as a custom error or allow it to propagate
        raise RuntimeError(f"Failed to build scorecard data: {e}") from e
