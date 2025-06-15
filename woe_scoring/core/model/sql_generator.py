from typing import Dict, List, Any # 'Any' for encoder type for now

# Assuming MISSING_PLACEHOLDER_FLOAT and MISSING_PLACEHOLDER_OBJECT might be needed
# if woe_dict['missing_bin'] logic relies on comparing with these.
# For now, the logic seems to be string "first"/"last" or direct value comparison.
# Let's add them if they become necessary for consistency with binning parts.
# from ..binning.functions import MISSING_PLACEHOLDER_OBJECT, MISSING_PLACEHOLDER_FLOAT
# For now, not importing them as the SQL generation seems to rely on bin['bin'][0] for missing value representation.

def _generate_categorical_case_statements(var_name: str, woe_transformation_dict: Dict) -> List[str]:
    """
    Generates SQL CASE statements for a categorical feature's WOE transformation.

    Args:
        var_name: The original variable name (e.g., "WOE_FeatureA").
        woe_transformation_dict: The dictionary containing WOE transformation details for this feature.
                                 Expected keys: 'type_feature', 'missing_bin', and the feature name itself
                                 containing a list of bin dictionaries.
    Returns:
        A list of SQL CASE statement strings.
    """
    original_feature_name = var_name.replace("WOE_", "")
    sql_case_parts = []

    # Bins are expected to be a list of dicts, where each dict has 'bin' (list of categories) and 'woe'.
    if not isinstance(woe_transformation_dict, dict):
        raise TypeError(f"woe_transformation_dict for {original_feature_name} must be a dictionary.")

    feature_bins_info = woe_transformation_dict.get(original_feature_name)
    if feature_bins_info is None:
        # Log: print(f"Warning: No binning information found for feature '{original_feature_name}' in woe_transformation_dict.")
        return sql_case_parts # Return empty if no info

    if not isinstance(feature_bins_info, list):
        raise TypeError(f"Binning information for feature '{original_feature_name}' must be a list of bin dicts.")

    for bin_info in feature_bins_info:
        if not isinstance(bin_info, dict):
            # Log: print(f"Warning: bin_info for {original_feature_name} is not a dict: {bin_info}. Skipping.")
            continue

        categories_in_bin = bin_info.get('bin')
        woe_value = bin_info.get('woe')

        if categories_in_bin is None or woe_value is None:
            # Log: print(f"Warning: 'bin' or 'woe' key missing in bin_info for {original_feature_name}: {bin_info}. Skipping.")
            continue

        if not isinstance(categories_in_bin, list):
            # Log: print(f"Warning: 'bin' value is not a list in bin_info for {original_feature_name}: {bin_info}. Skipping.")
            continue

        # Format categories for SQL 'IN' clause: ('cat1', 'cat2', ...)
        # Handle numeric categories vs string categories appropriately for quoting.
        formatted_categories = []
        for cat in categories_in_bin:
            if isinstance(cat, str):
                # Replace single quotes in string categories to avoid SQL injection/error
                formatted_categories.append("'{str(cat).replace(\"'\", \"''\")}'")
            else: # Numeric, boolean, etc.
                formatted_categories.append(str(cat))


        if not formatted_categories: # Skip if bin is empty for some reason
            continue

        categories_sql_tuple = f"({', '.join(formatted_categories)})"
        sql_case_parts.append(f" WHEN {original_feature_name} IN {categories_sql_tuple} THEN {woe_value:.8f}")

    # Handling missing values based on 'missing_bin' strategy
    missing_bin_strategy = woe_transformation_dict.get("missing_bin")
    if missing_bin_strategy and feature_bins_info: # Ensure feature_bins_info is not empty
        if missing_bin_strategy == "first":
            first_bin_woe = feature_bins_info[0].get('woe')
            if first_bin_woe is not None:
                sql_case_parts.append(f" WHEN {original_feature_name} IS NULL THEN {first_bin_woe:.8f}")
        elif missing_bin_strategy == "last":
            last_bin_woe = feature_bins_info[-1].get('woe')
            if last_bin_woe is not None:
                sql_case_parts.append(f" WHEN {original_feature_name} IS NULL THEN {last_bin_woe:.8f}")

    # If no specific missing handling, or if an ELSE is desired for unmapped values:
    # else:
    #   sql_case_parts.append(f" ELSE NULL") # Or some default WOE

    return sql_case_parts

def _generate_numeric_case_statements(var_name: str, woe_transformation_dict: Dict) -> List[str]:
    """
    Generates SQL CASE statements for a numeric feature's WOE transformation.

    Args:
        var_name: The original variable name (e.g., "WOE_FeatureB").
        woe_transformation_dict: The dictionary containing WOE transformation details.
    Returns:
        A list of SQL CASE statement strings.
    """
    original_feature_name = var_name.replace("WOE_", "")
    sql_case_parts = []

    if not isinstance(woe_transformation_dict, dict):
        raise TypeError(f"woe_transformation_dict for {original_feature_name} must be a dictionary.")

    feature_bins_info = woe_transformation_dict.get(original_feature_name)
    if feature_bins_info is None:
        # Log: print(f"Warning: No binning information found for feature '{original_feature_name}' in woe_transformation_dict.")
        return sql_case_parts

    if not isinstance(feature_bins_info, list):
        raise TypeError(f"Binning information for feature '{original_feature_name}' must be a list of bin dicts.")

    # Handle missing values first, if specified by 'missing_bin'
    missing_bin_strategy = woe_transformation_dict.get("missing_bin")
    if missing_bin_strategy and feature_bins_info: # Ensure feature_bins_info is not empty
        if missing_bin_strategy == "first":
            first_bin_woe = feature_bins_info[0].get('woe')
            if first_bin_woe is not None:
                sql_case_parts.append(f" WHEN {original_feature_name} IS NULL THEN {first_bin_woe:.8f}")
        elif missing_bin_strategy == "last":
            last_bin_woe = feature_bins_info[-1].get('woe')
            if last_bin_woe is not None:
                sql_case_parts.append(f" WHEN {original_feature_name} IS NULL THEN {last_bin_woe:.8f}")

    for i, bin_info in enumerate(feature_bins_info):
        if not isinstance(bin_info, dict):
            # Log: print(f"Warning: bin_info for {original_feature_name} is not a dict: {bin_info}. Skipping.")
            continue

        bin_boundaries = bin_info.get('bin')
        woe_value = bin_info.get('woe')

        if bin_boundaries is None or woe_value is None:
            # Log: print(f"Warning: 'bin' or 'woe' key missing in bin_info for {original_feature_name}: {bin_info}. Skipping.")
            continue

        if not (isinstance(bin_boundaries, list) and len(bin_boundaries) == 2):
            # Log: print(f"Warning: Numeric 'bin' value is not a list of two elements for {original_feature_name}: {bin_boundaries}. Skipping.")
            continue

        lower_bound, upper_bound = bin_boundaries[0], bin_boundaries[1]

        condition = ""
        # Ensure bounds are numeric or np.inf
        if not (isinstance(lower_bound, (int, float, np.number)) and isinstance(upper_bound, (int, float, np.number))):
             # Log: print(f"Warning: Non-numeric bin boundaries for {original_feature_name}: {bin_boundaries}. Skipping.")
            continue

        if lower_bound == -np.inf and upper_bound == np.inf:
             condition = "1=1" # This bin covers all non-null values if no other condition matched
        elif lower_bound == -np.inf:
            condition = f" {original_feature_name} < {float(upper_bound):.8f}"
        elif upper_bound == np.inf:
            condition = f" {original_feature_name} >= {float(lower_bound):.8f}"
        else:
            condition = f" {original_feature_name} >= {float(lower_bound):.8f} AND {original_feature_name} < {float(upper_bound):.8f}"

        sql_case_parts.append(f" WHEN{condition} THEN {woe_value:.8f}")

    # Add a final ELSE NULL for values that might not fit any bin (e.g. if IS NULL not handled and value is NULL)
    # or if there are gaps in bin definitions.
    # sql_case_parts.append(" ELSE NULL") # Or a default WOE if appropriate

    return sql_case_parts

def _finalize_sql_query(woe_feature_names: List[str], coefficients: List[float], intercept: float) -> List[str]:
    """
    Generates the final parts of the SQL query for calculating probability (PD).

    Args:
        woe_feature_names: List of WOE-transformed feature names (e.g., "WOE_FeatureA").
        coefficients: List of model coefficients for each WOE feature.
        intercept: The model intercept.
    Returns:
        A list of SQL query string parts.
    """
    if not (len(woe_feature_names) == len(coefficients)):
        raise ValueError("Length of woe_feature_names and coefficients must match.")

    # Assuming 'input_table' is the source table name, to be defined by the user or context.
    # The CTE 'a' will select original features and generate WOE columns.
    # This function generates the part of SQL that consumes the WOE columns from CTE 'a'.

    sql_calc_parts = [
        " FROM a) -- 'a' is the CTE with original features and generated WOE columns\n",
        ", probability_calculation AS (\n",
        "  SELECT a.* -- Select all columns from 'a' (original + WOE features)\n",
        f"  , 1.0 / (1.0 + EXP(-({intercept:.8f} -- Intercept\n"
    ]

    for i, woe_feature_name in enumerate(woe_feature_names):
        sql_calc_parts.append(f"    + ({coefficients[i]:.8f} * a.{woe_feature_name}) -- Coeff * {woe_feature_name}\n")

    sql_calc_parts.extend([
        "  ))) AS PD -- Probability of Default\n",
        "  FROM a\n",
        ") \n",
        "SELECT * FROM probability_calculation;" # Select all from final CTE
    ])
    return sql_calc_parts


def generate_sql_query(
    woe_encoder_info: Any, # Contains all woe_iv_dicts, type_feature, missing_bin strategy per feature
    woe_feature_names: List[str], # List of WOE-transformed feature names, e.g., "WOE_FeatureA"
    model_coefficients: List[float], # Coefficients from the logistic regression model, corresponding to woe_feature_names
    model_intercept: float
    ) -> str:
    """
    Generates a complete SQL query for scoring based on WOE transformations and a logistic regression model.

    Args:
        woe_encoder_info: An object or dict that contains all necessary WOE transformation rules.
                          Typically, this would be the `encoder.woe_iv_dict` from the original structure,
                          which is a list of dictionaries. Each dictionary is for one feature and
                          contains its WOE bins, type ('cat'/'num'), and missing_bin strategy.
        woe_feature_names: List of feature names after WOE transformation (e.g., "WOE_Age_binned").
                           These should match the features used in the model.
        model_coefficients: Coefficients from the logistic regression model.
        model_intercept: Intercept from the logistic regression model.

    Returns:
        A string containing the SQL query.
    """
    sql_parts = ["WITH a AS (SELECT"]

    # Extract original feature names needed for the SELECT statement from input_table
    original_feature_names_to_select = sorted(list(set([var.replace("WOE_", "") for var in woe_feature_names])))
    sql_parts.append(" " + ", ".join(original_feature_names_to_select))
    sql_parts.append("\n") # Newline after selected original features

    if not isinstance(woe_encoder_info, list):
        raise TypeError("woe_encoder_info is expected to be a list of rule dictionaries.")
    if not (len(woe_feature_names) == len(model_coefficients)):
        raise ValueError("Length of woe_feature_names and model_coefficients must match.")

    # Generate CASE statements for each WOE feature
    for woe_var_name in woe_feature_names:
        original_var_name = woe_var_name.replace("WOE_", "")

        feature_woe_rules = None
        for ruleset_for_one_feature in woe_encoder_info:
            if not isinstance(ruleset_for_one_feature, dict):
                # Log: print(f"Warning: Item in woe_encoder_info is not a dict: {ruleset_for_one_feature}. Skipping.")
                continue
            if original_var_name in ruleset_for_one_feature:
                feature_woe_rules = ruleset_for_one_feature
                break

        if feature_woe_rules is None:
            # Log error: print(f"Critical: No WOE rules found for original feature '{original_var_name}' (derived from '{woe_var_name}'). SQL will be incomplete.")
            # Consider raising an error to halt SQL generation if a feature's rules are missing.
            raise ValueError(f"Missing WOE rules for feature: {original_var_name} (from {woe_var_name})")

        sql_parts.append(f"  , CASE -- Generating WOE for {original_var_name} AS {woe_var_name}\n")

        feature_type = feature_woe_rules.get("type_feature")
        if feature_type is None:
            raise ValueError(f"'type_feature' missing in WOE rules for feature: {original_var_name}")

        if feature_type == "cat":
            sql_parts.extend(_generate_categorical_case_statements(woe_var_name, feature_woe_rules))
        elif feature_type == "num":
            sql_parts.extend(_generate_numeric_case_statements(woe_var_name, feature_woe_rules))
        else:
            # Log warning: print(f"Warning: Unknown feature type '{feature_type}' for {original_var_name}. WOE column will be NULL.")
            sql_parts.append(f"    -- Unknown feature type '{feature_type}' for {original_var_name}\n")
            sql_parts.append("    ELSE NULL -- Default for unknown type\n")

        sql_parts.append(f"  END AS {woe_var_name}\n")

    # The FROM clause should specify the source table name, which is context-dependent.
    # The _finalize_sql_query function assumes the CTE 'a' is already defined up to this point.
    # The initial " FROM input_table)" part has been moved into _finalize_sql_query for better structure.
    # We need to ensure the initial SELECT is FROM a user-specified table.
    # For now, the CTE 'a' selects from 'input_table' as a placeholder.
    # A better design might be to pass 'input_table_name' as an argument.
    sql_parts.append(" FROM input_table) -- Replace 'input_table' with your actual source table name\n")


    sql_parts.extend(_finalize_sql_query(woe_feature_names, model_coefficients, model_intercept))
    return "".join(sql_parts)

import numpy as np # Added for -np.inf, np.inf usage
