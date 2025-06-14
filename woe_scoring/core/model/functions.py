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

from .model import Model # Assuming Model class is used by calc_model_results
# Import new builder modules
from . import scorecard_builder
import logging # Add logging
from . import excel_builder

logger = logging.getLogger(__name__) # Module-level logger

# Functions that were not part of the static classes, or orchestrate them.

def save_scorecard(
    woe_feature_names_in_model: List[str], # Actual features in model, e.g. ["WOE_Age", "WOE_Income"]
    woe_encoder_rules: List[Dict[str, Any]], # Contains all WOE rules (e.g., encoder.woe_iv_dict)
    model_coeff_pvalue_df: pd.DataFrame, # DataFrame from calc_model_results (includes 'const')
    base_points: int,
    odds: int, # Example: 50 for 50:1 odds
    points_to_double_odds: int, # PDO
    output_path: str
    ) -> None:
    """
    Calculates scorecard statistics and saves the complete scorecard to an Excel file.

    Args:
        woe_feature_names_in_model: List of WOE-transformed feature names used in the model.
        woe_encoder_rules: The collection of WOE transformation rules for all features.
        model_coeff_pvalue_df: DataFrame containing model coefficients and p-values for features and 'const'.
                               Expected columns: first column for names (e.g. 'const', 'WOE_Age'), 'coef', 'P>|z|'.
        base_points: Target score for the specified odds.
        odds: The odds for which the base_points are set (e.g., 50 for 50:1).
        points_to_double_odds: Points needed to double the odds.
        output_path: Directory path to save the "Scorecard.xlsx" file.
    """
    # Calculate factor and offset for scorecard point calculation
    # factor = pdo / ln(2)
    # offset = score - factor * ln(odds)
    factor = points_to_double_odds / np.log(2)
    offset = base_points - factor * np.log(odds) # np.log is natural log (ln)
    logger.info(f"Scorecard calculation: factor={factor:.4f}, offset={offset:.4f}")

    try:
        logger.info("Building scorecard data.")
        list_of_feature_scorecard_dfs = scorecard_builder.build_scorecard_data(
            woe_feature_names_in_model=woe_feature_names_in_model,
            woe_encoder_rules=woe_encoder_rules,
            model_results_df=model_coeff_pvalue_df,
            factor=factor,
            offset=offset
        )

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
    Calculates a summary DataFrame of model results (coefficients and p-values).

    Args:
        model_object: An instance of the Model class.

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
