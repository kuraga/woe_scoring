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
from . import excel_builder


# Functions that were not part of the static classes, or orchestrate them.

def save_scorecard(
    woe_feature_names_in_model: List[str], # Actual features in model, e.g. ["WOE_Age", "WOE_Income"]
    woe_encoder_rules: Any, # Contains all WOE rules (e.g., encoder.woe_iv_dict)
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

    try:
        # Build the list of DataFrames, one for each feature's scorecard contribution
        list_of_feature_scorecard_dfs = scorecard_builder.build_scorecard_data(
            woe_feature_names_in_model=woe_feature_names_in_model,
            woe_encoder_rules=woe_encoder_rules,
            model_results_df=model_coeff_pvalue_df, # Pass the df with coeffs and pvalues
            factor=factor,
            offset=offset
        )

        # Define the full path for the Excel file
        excel_file_path = os.path.join(output_path, "Scorecard.xlsx")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        with pd.ExcelWriter(excel_file_path, engine="xlsxwriter") as writer:
            excel_builder.build_excel_scorecard_sheet(
                all_feature_scorecard_data=list_of_feature_scorecard_dfs,
                excel_writer=writer
                # Default chart width/height and positions will be used
            )
        print(f"Scorecard saved to {excel_file_path}")

    except Exception as e:
        # Consider more specific error handling or logging
        print(f"Error saving scorecard: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error during development


def calc_model_results(model_object: Model) -> pd.DataFrame:
    """
    Calculates a summary DataFrame of model results (coefficients and p-values).

    Args:
        model_object: An instance of the Model class (or any object with .feature_names_,
                      .intercept_, .coef_, .pvalues_ attributes).

    Returns:
        A pandas DataFrame with columns like 'index' (feature name or 'const'), 'coef', 'P>|z|'.
    """
    # Assumes model_object.feature_names_ are the WOE_transformed names.
    # Assumes model_object.coef_ and model_object.pvalues_ correspond to these feature_names_.
    feature_names_for_df = ['const'] + [name for name in model_object.feature_names_]
    coefficients_for_df = [model_object.intercept_] + list(model_object.coef_)
    pvalues_for_df = [0.0] + list(model_object.pvalues_) # p-value for const is often set to 0 or not applicable

    results_df = pd.DataFrame({
        # The first column name is not strictly 'index', but will be used by other functions to find 'const'
        'Feature': feature_names_for_df,
        'coef': coefficients_for_df,
        'P>|z|': pvalues_for_df
    })
    # No reset_index(drop=True) needed if DataFrame is constructed this way.
    return results_df
