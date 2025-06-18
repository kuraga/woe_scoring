import os
from functools import lru_cache, partial
from itertools import combinations
from typing import Dict, List, Union, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .model import Model


def calculate_gini_score(
        data: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        feature: str,
        random_state: int,
        class_weight: str,
        cv: int,
        scoring: str,
        n_jobs: int
) -> float:
    """
    Calculate the Gini score for a given feature using Logistic Regression.

    Args:
        data: A pandas DataFrame or numpy array containing the feature and target data.
        target: A pandas Series or numpy array containing the target data.
        feature: A string representing the name of the feature to calculate the Gini score for.
        random_state: An integer representing the random state for the Logistic Regression estimator.
        class_weight: A string or dictionary representing the class weight for the Logistic Regression estimator.
        cv: An integer representing the number of cross-validation folds to use.
        scoring: A string representing the scoring metric to use for cross-validation.
        n_jobs: An integer representing the number of parallel jobs to run during cross-validation.

    Returns:
        A float representing the Gini score for the given feature.
    """
    # Pre-reshape feature data for improved performance
    X = data[feature].values.reshape(-1, 1)

    # Create model with optimal parameters for fast convergence
    estimator = LogisticRegression(
        random_state=random_state,
        class_weight=class_weight,
        max_iter=1000,
        n_jobs=1,  # Use 1 for the inner job to avoid nested parallelism
        warm_start=True,
        solver='liblinear'  # Faster for small datasets/single feature
    )

    # Calculate cross-validation scores
    scores = cross_val_score(
        estimator=estimator,
        X=X,
        y=target,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    return (np.mean(scores) * 2 - 1) * 100


def calc_features_gini_quality(
        data: Union[pd.DataFrame, np.ndarray],
        target: Union[pd.Series, np.ndarray],
        feature_names: List[str],
        random_state: int,
        class_weight: str,
        cv: int,
        scoring: str,
        n_jobs: int,
) -> Dict[str, float]:
    """
    Calculates scorecard statistics and saves the complete scorecard to an Excel file.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): The dataset from which to calculate feature quality.
        target (Union[pd.Series, np.ndarray]): The target variable.
        feature_names (List[str]): The names of the features to be evaluated.
        random_state (int): Seed used by the random number generator.
        class_weight (str): Weights associated with classes in the form of a dictionary.
        cv (int): Number of folds used for cross-validation.
        scoring (str): The evaluation metric to score predictions.
        n_jobs (int): Number of CPU cores used for parallelization.

    Returns:
        Dict: A dictionary containing the calculated Gini quality of each feature.
    """
    # Ensure target is numpy array for consistency
    target_array = target.values if hasattr(target, 'values') else np.array(target)

    # Use parallel processing for faster computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_gini_score)(
            data=data,
            target=target_array,
            feature=feature_name,
            random_state=random_state,
            class_weight=class_weight,
            cv=cv,
            scoring=scoring,
            n_jobs=1,  # Use single process within each job to avoid nested parallelism
        )
        for feature_name in feature_names
    )

    return dict(zip(feature_names, results))


def check_features_gini_threshold(
        feature_names: List[str],
        features_gini_scores: Dict[str, float],
        gini_threshold: float,
) -> List[str]:
    """
    Check for the feature names whose Gini impurity is greater than or equal to a given threshold.

    :param feature_names: A list of feature names.
    :type feature_names: List[str]
    :param features_gini_scores: A dictionary of feature names with their corresponding Gini impurity.
    :type features_gini_scores: Dict[str, float]
    :param gini_threshold: The minimum Gini impurity threshold to filter the features.
    :type gini_threshold: float
    :return: A filtered list of feature names whose Gini impurity is greater than or equal to the threshold.
    :rtype: List[str]
    """
    return [feature_name for feature_name in feature_names if features_gini_scores[feature_name] >= gini_threshold]


def check_correlation_threshold(
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: List[str],
        features_gini_scores: Dict[str, float],
        corr_threshold: float
) -> List[str]:
    """
    Check correlation matrix for features in given data, and return only uncorrelated
    features with respect to given correlation threshold.

    :param data: The data to check correlation matrix.
    :type data: Union[pd.DataFrame, np.ndarray]
    :param feature_names: The names of the features in the data.
    :type feature_names: List[str]
    :param features_gini_scores: The Gini indices of the features.
    :type features_gini_scores: Dict[str, float]
    :param corr_threshold: The correlation threshold to check against.
    :type corr_threshold: float
    :return: The uncorrelated feature names.
    :rtype: List[str]
    """
    # Handle empty feature list
    if not feature_names:
        return []

    # Calculate correlation matrix only once
    correlation_matrix = data[feature_names].corr().abs()

    # More efficient algorithm to remove correlated features
    # Start with all features
    uncorrelated_features = set(feature_names)

    # Pre-sort features by Gini score (higher score first)
    sorted_features = sorted(feature_names, key=lambda x: features_gini_scores.get(x, 0), reverse=True)

    # For each feature (in order of decreasing importance)
    for i, feature_a in enumerate(sorted_features):
        # If feature already removed, skip
        if feature_a not in uncorrelated_features:
            continue

        # Compare with remaining features
        for feature_b in sorted_features[i+1:]:
            if feature_b not in uncorrelated_features:
                continue

            # If correlation exceeds threshold, remove the less important feature
            if abs(correlation_matrix.loc[feature_a, feature_b]) >= corr_threshold:
                uncorrelated_features.discard(feature_b)

    return list(uncorrelated_features)


def check_min_pct_group(
        data: Union[pd.DataFrame, np.ndarray],
        feature_names: List[str],
        min_pct_group: float,
) -> List[str]:
    """
    Check if a feature has a minimum percentage of values below a threshold.

    Args:
        data: DataFrame or numpy array.
        feature_names: List of features.
        min_pct_group: Minimum percentage of values below a threshold.

    Returns:
        List of features with a minimum percentage of values above or equal to the threshold.
    """
    # More efficient implementation with early stopping
    valid_features = []

    for feature_name in feature_names:
        # Calculate value counts and find minimum percentage
        value_counts = data[feature_name].value_counts(normalize=True)

        # Skip expensive min() calculation if possible
        if len(value_counts) == 0:
            continue
        elif len(value_counts) == 1:
            # Single value feature - keep it as it has 100% in one group
            valid_features.append(feature_name)
        else:
            # Check if smallest percentage is above threshold
            if value_counts.min() >= min_pct_group:
                valid_features.append(feature_name)

    return valid_features


def find_bad_features(model: Model) -> List[str]:
    """Find features with high p-values and positive sign.
    Args:
        model: Model.
    Returns:
        List of features with high p-values or positive coefficients.
    """
    bad_features = []

    for i, feature in enumerate(model.feature_names_):
        # Check if p-value is too high (not statistically significant)
        # or if coefficient is positive (for binary classification with 0/1 target)
        if model.pvalues_[i] > 0.05 or model.coef_[i] > 0:
            bad_features.append(feature)

    return bad_features


def calc_iv_dict(data: pd.DataFrame, target: np.ndarray, feature: str) -> Dict:
    """Calculate the information value (IV) of a categorical feature.

    Args:
        data: A pandas DataFrame containing the feature and target columns.
        target: A numpy array of binary labels (0 for good, 1 for bad).
        feature: A string with the name of the categorical feature.

    Returns:
        A dictionary with the feature name as key and the IV as value.
    """

    values = data[feature].values
    unique_values, value_counts = np.unique(values, return_counts=True)
    bad = np.zeros_like(unique_values)
    good = np.zeros_like(unique_values)
    for i, value in enumerate(unique_values):
        bad[i] = target[values == value].sum()
        good[i] = value_counts[i] - bad[i]
    all_bad = target.sum()
    all_good = len(target) - all_bad
    iv = ((good / all_good) - (bad / all_bad)) * unique_values
    return {feature: iv.sum()}


def save_reports(
        model: sm.Logit,
        path: str = os.getcwd()
) -> None:
    """Save model reports.
    Args:
        model: Model.
        path: Path to save reports."""

    try:
        with open(
                os.path.join(path, "model_summary.txt"), "w"
        ) as outfile:
            outfile.write(model.summary().as_text())

        with open(
                os.path.join(path, "model_wald.txt"), "w"
        ) as outfile:
            model.wald_test_terms().summary_frame().to_string(outfile)
    except Exception as e:
        print(f"Problem with saving: {e}")


def generate_sql(
        encoder, feature_names: List[str], coef: List[float], intercept: float,
) -> str:
    """
    Generate SQL query for model deployment based on fitted model.

    Args:
        encoder: WOE encoder used to transform features
        feature_names: List of feature names in the model
        coef: List of coefficient values
        intercept: Intercept value

    Returns:
        str: SQL query for model scoring
    """
    # Strip WOE_ prefix from original feature names
    base_features = [var.replace("WOE_", "") for var in feature_names]

    # Initialize SQL query parts
    sql = [
        "with a as (SELECT ",
        ",".join(base_features),
        "",
    ]

    # Process each feature's transformation
    for var in feature_names:
        base_var = var.replace("WOE_", "")

        # Find corresponding encoder dictionary entry
        woe_dict_entry = None
        for entry in encoder.woe_iv_dict:
            if list(entry.keys())[0] == base_var:
                woe_dict_entry = entry
                break

        if not woe_dict_entry:
            continue  # Skip if not found

        # Start CASE statement
        sql.append(", CASE")

        # Handle categorical vs numerical features differently
        if woe_dict_entry["type_feature"] == "cat":
            # Categorical feature binning
            for bin_info in woe_dict_entry[base_var]:
                bin_str = str(bin_info["bin"]).replace("[", "(").replace("]", ")") \
                                             .replace(", -1", "").replace(", Missing", "")
                sql.append(f" WHEN {base_var} in {bin_str} THEN {bin_info['woe']}")

            # Handle missing values
            if woe_dict_entry["missing_bin"] == "first":
                first_bin_woe = woe_dict_entry[base_var][0]['woe']
                sql.append(f" WHEN {base_var} IS NULL THEN {first_bin_woe}")
                sql.append(f" ELSE {first_bin_woe}")
            elif woe_dict_entry["missing_bin"] == "last":
                last_bin_woe = woe_dict_entry[base_var][-1]['woe']
                sql.append(f" WHEN {base_var} IS NULL THEN {last_bin_woe}")
                sql.append(f" ELSE {last_bin_woe}")
        else:
            # Numerical feature binning

            # Handle NULL values first
            if woe_dict_entry["missing_bin"] == "first":
                sql.append(f" WHEN {base_var} IS NULL THEN {woe_dict_entry[base_var][0]['woe']}")
            elif woe_dict_entry["missing_bin"] == "last":
                sql.append(f" WHEN {base_var} IS NULL THEN {woe_dict_entry[base_var][-1]['woe']}")

            # Handle numeric bins
            for n, bin_info in enumerate(woe_dict_entry[base_var]):
                if n == 0:
                    sql.append(f" WHEN {base_var} < {bin_info['bin'][1]} THEN {bin_info['woe']}")
                elif n == len(woe_dict_entry[base_var]) - 1:
                    sql.append(f" WHEN {base_var} >= {bin_info['bin'][0]} THEN {bin_info['woe']}")
                else:
                    sql.append(
                        f" WHEN {base_var} >= {bin_info['bin'][0]} AND {base_var} < {bin_info['bin'][1]} "
                        f"THEN {bin_info['woe']}"
                    )

        # Close CASE statement
        sql.append(f" END AS {var}")

    # Add model formula
    sql.extend([
        " FROM )",
        ", b as (",
        "SELECT a.*",
        f", REPLACE(1 / (1 + EXP(-({intercept}"
    ])

    # Add feature coefficients
    for idx, feature in enumerate(feature_names):
        sql.append(f" + ({coef[idx]} * a.{feature})")

    # Finish query
    sql.extend([
        "))), ',', '.') as PD",
        " FROM a) ",
        "SELECT * FROM b"
    ])

    return "".join(sql)

        excel_file_path = os.path.join(output_path, "Scorecard.xlsx")
        if not os.path.exists(output_path):
            logger.info(f"Creating output directory for scorecard: {output_path}")
            os.makedirs(output_path, exist_ok=True)

        logger.info(f"Saving scorecard to Excel file: {excel_file_path}")
        with pd.ExcelWriter(excel_file_path, engine="xlsxwriter") as writer:
            excel_builder.build_excel_scorecard_sheet(
                all_feature_scorecard_data=list_of_feature_scorecard_dfs,
                excel_writer=writer
            )
        logger.info(f"Scorecard saved successfully to {excel_file_path}")

    except Exception as e:
        logger.error(f"Error saving scorecard: {e}", exc_info=True)
        # Re-raise the exception if it's critical for the calling process
        # raise


def calc_model_results(model_object: Model) -> pd.DataFrame:
    """
    result_dict = {
        "feature": [],
        "coef": [],
        "pvalue": [],
        "bin": [],
        "WOE": [],
        "IV": [],
        "percent_of_population": [],
        "total": [],
        "event_cnt": [],
        "non_event_cnt": [],
        "event_rate": [],
        "score_ball": [],
    }

    woe_iv_dict = encoder.woe_iv_dict
    intercept = model_results.iloc[0, 1]
    n_features = len(feature_names)

    if idx < 1:
        _update_result_dict(result_dict, feature, model_results, idx)
        for key, value in result_dict.items():
            if key not in ["feature", "coef", "pvalue"]:
                value.append("-")
    else:
        for woe_iv in woe_iv_dict:
            if list(woe_iv.keys())[0] == feature.replace("WOE_", ""):
                feature_woe_iv = woe_iv[feature.replace("WOE_", "")]
                for bin_info in feature_woe_iv:
                    _update_result_dict(result_dict, feature, model_results, idx)
                    bin_values = bin_info["bin"]
                    bin_values_str = [
                        str(val).replace("-1", "missing") if val == -1 else val
                        for val in bin_values
                    ]
                    result_dict["bin"].append(bin_values_str)
                    result_dict["WOE"].append(bin_info["woe"])
                    result_dict["IV"].append(bin_info["iv"])
                    result_dict["percent_of_population"].append(bin_info["pct"])
                    result_dict["total"].append(bin_info["total"])
                    result_dict["event_cnt"].append(bin_info["bad"])
                    result_dict["non_event_cnt"].append(bin_info["good"])
                    result_dict["event_rate"].append(bin_info["bad_rate"])
                    result_dict["score_ball"].append(
                        _calc_score_points(
                            woe=result_dict["WOE"][-1],
                            coef=result_dict["coef"][-1],
                            intercept=intercept,
                            factor=factor,
                            offset=offset,
                            n_features=n_features,
                        )
                    )

    return pd.DataFrame.from_dict(result_dict)


def _update_result_dict(result_dict, feature, model_results, idx) -> None:
    """Update result dict with feature information.
    Args:
        result_dict: Dictionary to update with feature information.
        feature: Feature name.
        model_results: DataFrame with model coefficient results.
        idx: Index in model_results.
    Returns:
        None: Updates result_dict in-place.
    """
    # Extract base feature name by removing WOE_ prefix
    feature_name = feature.replace("WOE_", "")

    # Add feature information to the result dictionary
    result_dict["feature"].append(feature_name)
    result_dict["coef"].append(model_results.loc[idx, "coef"])
    result_dict["pvalue"].append(model_results.loc[idx, "P>|z|"])


def _calc_stats(
        feature_names: List[str],
        encoder,
        model_results,
        factor: float,
        offset: float,
) -> List[pd.DataFrame]:
    """Calculate feature statistics for reporting.
    Args:
        feature_names: List of feature names in the model.
        encoder: WOE encoder used for transformations.
        model_results: Model coefficient results.
        factor: Scaling factor for points calculation.
        offset: Offset value for points calculation.
    Returns:
        List of DataFrames with feature statistics.
    """
    # Extract features to process from model results
    features_to_process = model_results.iloc[:, 0]

    # Use parallel processing with explicit parameters for efficiency
    return Parallel(n_jobs=-1, backend="threading", prefer="threads")(
        delayed(_calc_stats_for_feature)(
            idx=idx,
            feature=feature,
            feature_names=feature_names,
            encoder=encoder,
            model_results=model_results,
            factor=factor,
            offset=offset
        ).rename(feature.replace("WOE_", ""))
        for idx, feature in enumerate(features_to_process)
    )


    Returns:
        A pandas DataFrame with columns 'Feature', 'coef', 'P>|z|'.
    """
    logger.info("Calculating model results (coefficients and p-values).")

    # Ensure feature_names_, coef_, and pvalues_ are available and aligned
    if not hasattr(model_object, 'feature_names_') or \
       not hasattr(model_object, 'coef_') or \
       not hasattr(model_object, 'pvalues_') or \
       not hasattr(model_object, 'intercept_'):
        logger.error("Model object is missing required attributes (feature_names_, coef_, pvalues_, intercept_).")
        raise AttributeError("Model object is missing required attributes for calculating results.")

    if len(model_object.feature_names_) != len(model_object.coef_):
        logger.error("Length mismatch between feature_names_ and coef_.")
        raise ValueError("Model's feature_names_ and coef_ have inconsistent lengths.")
    if len(model_object.feature_names_) != len(model_object.pvalues_):
        logger.error("Length mismatch between feature_names_ and pvalues_.")
        raise ValueError("Model's feature_names_ and pvalues_ have inconsistent lengths.")

    feature_names_for_df = ['const'] + list(model_object.feature_names_)
    coefficients_for_df = [model_object.intercept_] + list(model_object.coef_)
    # Assuming pvalues_ from model_object corresponds to features; p-value for const might not be directly available
    # or meaningful in the same way depending on the model type (e.g. sklearn LogisticRegression).
    # For statsmodels, intercept p-value is available. Model class should abstract this.
    # If model_object.intercept_pvalue_ exists, use it. Otherwise, default (e.g., 0.0 or NaN).
    intercept_pvalue = getattr(model_object, 'intercept_pvalue_', 0.0) # Default if not available
    logger.debug(f"Using intercept p-value: {intercept_pvalue}")
    pvalues_for_df = [intercept_pvalue] + list(model_object.pvalues_)

    results_df = pd.DataFrame({
        'Feature': feature_names_for_df,
        'coef': coefficients_for_df,
        'P>|z|': pvalues_for_df
    })
    logger.info("Model results calculated successfully.")
    return results_df
