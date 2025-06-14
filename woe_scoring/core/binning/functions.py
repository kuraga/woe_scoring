from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import chisquare


@dataclass(frozen=True)
class BadRates:
    bad: float
    total: int
    pct: float
    bad_rate: float
    woe: float
    iv: float
    bin: Union[List, float]


@dataclass(frozen=True)
class BinningResult:
    bad_rates: List[BadRates]
    missing_bin: Optional[str] = None
    type_feature: Optional[str] = None
    bins: List = field(default_factory=list)


def _chi2(bad_rates: List[BadRates], overall_rate: float) -> float:
    """Calculate chi-square statistic"""
    f_obs = np.array([b.bad for b in bad_rates])
    f_exp = f_obs * overall_rate
    return chisquare(f_obs=f_obs, f_exp=f_exp)[0]


@lru_cache(maxsize=128)
def _check_diff_woe(bad_rates: Tuple[BadRates], diff_woe_threshold: float) -> Optional[int]:
    """Check if WOE differences are below threshold"""
    woe_delta = np.abs(np.diff([b.woe for b in bad_rates]))
    min_diff_woe = np.min(woe_delta)
    if min_diff_woe < diff_woe_threshold:
        return int(np.argmin(woe_delta))
    return None


def _mono_flags(bad_rates: List[BadRates]) -> bool:
    """Check if bad rates are monotonic"""
    bad_rate_arr = np.array([b.bad_rate for b in bad_rates])
    diffs = np.diff(bad_rate_arr)
    return np.all(diffs > 0) or np.all(diffs < 0)


def _find_index_of_diff_flag(bad_rates: List[BadRates]) -> int:
    """Find index where monotonicity changes"""
    bad_rate_diffs = np.diff([b.bad_rate for b in bad_rates])
    diffs_bool = bad_rate_diffs > 0
    return np.argmin(diffs_bool)


def _merge_bins_chi(x: np.ndarray, y: np.ndarray,
                   bad_rates: List[BadRates], bins: List) -> Tuple[List[BadRates], List]:
    """Merge bins using chi-square method"""
    idx = _find_index_of_diff_flag(bad_rates)
    if idx == 0:
        del bins[1]
    elif idx == len(bad_rates) - 2:
        del bins[len(bins) - 2]
    else:
        _extract_bin_by_chi2(bins, idx, x, y)

    result = _bin_bad_rates(x, y, bins)
    return result.bad_rates, result.bins


def _extract_bin_by_chi2(bins: List, idx: int, x: np.ndarray, y: np.ndarray) -> None:
    """Extract bin based on chi-square values"""
    temp_bins = bins.copy()
    del temp_bins[idx + 1]
    temp_result = _bin_bad_rates(x, y, temp_bins)
    chi_1 = _chi2(temp_result.bad_rates, temp_result.overall_rate or 0.0)

    temp_bins = bins.copy()
    del temp_bins[idx + 2]
    temp_result = _bin_bad_rates(x, y, temp_bins)
    chi_2 = _chi2(temp_result.bad_rates, temp_result.overall_rate or 0.0)

    if chi_1 < chi_2:
        del bins[idx + 1]
    else:
        del bins[idx + 2]


def _merge_bins_iv(x: np.ndarray, y: np.ndarray,
                  bad_rates: List[BadRates], bins: List) -> Tuple[List[BadRates], List]:
    """Merge bins using IV method"""
    idx = _find_index_of_diff_flag(bad_rates)
    if idx == 0:
        del bins[1]
    elif idx == len(bad_rates) - 2:
        del bins[len(bins) - 2]
    else:
        _extract_bin_by_iv(bins, idx, x, y)

    result = _bin_bad_rates(x, y, bins)
    return result.bad_rates, result.bins


def _extract_bin_by_iv(bins: List, idx: int, x: np.ndarray, y: np.ndarray) -> None:
    """Extract bin based on IV values"""
    temp_bins = bins.copy()
    del temp_bins[idx + 1]
    temp_result = _bin_bad_rates(x, y, temp_bins)
    iv_1 = sum(b.iv for b in temp_result.bad_rates)

    temp_bins = bins.copy()
    del temp_bins[idx + 2]
    temp_result = _bin_bad_rates(x, y, temp_bins)
    iv_2 = sum(b.iv for b in temp_result.bad_rates)

    if iv_1 > iv_2:
        del bins[idx + 1]
    else:
        del bins[idx + 2]


def _merge_bins_min_pct(x: np.ndarray, y: np.ndarray,
                       bad_rates: List[BadRates], bins: List,
                       cat: bool = False) -> Tuple[List[BadRates], List]:
    """Merge bins with minimum percentage"""
    idx = np.argmin([b.pct for b in bad_rates])

    if cat:
        if idx == 0:
            bins[idx + 1].extend(bins[idx])
        elif idx == len(bad_rates) - 1:
            bins[idx - 1].extend(bins[idx])
        elif bad_rates[idx - 1].pct < bad_rates[idx + 1].pct:
            bins[idx - 1].extend(bins[idx])
        else:
            bins[idx + 1].extend(bins[idx])
        del bins[idx]
    elif idx == 0:
        del bins[1]
    elif idx == len(bad_rates) - 1:
        del bins[len(bins) - 2]
    elif bad_rates[idx - 1].pct < bad_rates[idx + 1].pct:
        del bins[idx]
    else:
        del bins[idx + 1]

    result = _bin_bad_rates(x, y, bins, cat=cat)
    if cat:
        bins = [b.bin for b in result.bad_rates]
    return result.bad_rates, bins


@dataclass
class BinBadRatesResult:
    """Results container for bin bad rates calculation"""
    bad_rates: List[BadRates]
    overall_rate: Optional[float] = None
    bins: List = field(default_factory=list)


def _calc_stats(x: np.ndarray, y: np.ndarray, idx: int,
                all_bad: int, all_good: int, bins: List,
                cat: bool = False, refit_fl: bool = False) -> BadRates:
    """Calculate statistics for a single bin"""
    if refit_fl:
        value = bins[idx]
    else:
        value = bins[idx] if cat else [bins[idx], bins[idx + 1]]

    mask = ~pd.isna(x)
    x_not_na = x[mask]
    y_not_na = y[mask]

    if cat:
        bin_mask = pd.Series(x_not_na).isin(value)
    else:
        bin_mask = (x_not_na >= np.min(value)) & (x_not_na < np.max(value))

    total = bin_mask.sum()
    bad = y_not_na[bin_mask].sum()
    pct = bin_mask.mean()
    bad_rate = bad / total if total else 0
    good = total - bad

    # Smoothing for WOE calculation, applied before calculating term_good/term_bad
    # to ensure 'bad' and 'good' are floats if they become 0.5
    # This also means 'bad' in BadRates output can be float.
    # Original code added 0.5 if good or bad is zero.
    # Let's refine: if total is 0, then bad and good are 0. Adding 0.5 prevents log(0)/division by zero.
    # If total > 0, but bad or good is 0, also add 0.5.
    if total == 0 or bad == 0 or good == 0:
        good += 0.5
        bad += 0.5

    # Calculate terms for WOE and IV, preventing division by zero
    term_good = good / all_good if all_good > 0 else 0
    term_bad = bad / all_bad if all_bad > 0 else 0

    # Calculate WOE, preventing log(0)
    if term_good == 0 or term_bad == 0:
        # Define WOE for cases where one of the terms is zero.
        # This might indicate perfect separation or an empty bin after adjustments.
        # A large magnitude WOE is often used, or 0 if it implies no information.
        # For now, let's use a large value if one is zero and the other isn't,
        # and 0 if both are zero (e.g. after smoothing an empty bin).
        if term_good == 0 and term_bad == 0: # Both zero (e.g. empty bin, smoothed)
            woe = 0.0
        elif term_bad == 0: # Bad is zero (or very small), good is present
            woe = 7.0 # Represents very low bad rate (highly "good" bin)
        elif term_good == 0: # Good is zero (or very small), bad is present
            woe = -7.0 # Represents very high bad rate (highly "bad" bin)
        else: # Should not be reached if logic is correct, but as a fallback:
            woe = 0.0
    else:
        woe = np.log(term_good / term_bad)
        # Cap WOE to avoid extreme values if necessary, e.g. np.clip(woe, -7, 7)
        woe = np.clip(woe, -7, 7)


    iv = (term_good - term_bad) * woe

    return BadRates(
        bin=value,
        total=total, # Original total before smoothing
        bad=y_not_na[bin_mask].sum(), # Original bad count before smoothing
        pct=pct,
        bad_rate=bad_rate,
        woe=woe,
        iv=iv
    )


def _bin_bad_rates(x: np.ndarray, y: np.ndarray, bins: List,
                   cat: bool = False, refit_fl: bool = False) -> BinBadRatesResult:
    """Calculate bad rates for all bins"""
    all_bad_sum = int(y.sum())
    all_total_sum = len(y)
    all_good_sum = all_total_sum - all_bad_sum

    # Handle edge case: if all_bad_sum or all_good_sum is 0 for the entire dataset,
    # this can lead to issues in _calc_stats's WOE/IV if not handled.
    # _calc_stats has smoothing for individual bins, but global all_bad/all_good might also be zero.
    # If all_bad_sum is 0, all y are 0. If all_good_sum is 0, all y are 1.
    # This is a rare scenario for a typical binary target, but good to be aware of.
    # The division protection in _calc_stats (term_good/term_bad) should handle this.

    max_idx = len(bins) if cat or refit_fl else len(bins) - 1

    bad_rates = [
        _calc_stats(x, y, idx, all_bad_sum, all_good_sum, bins, cat, refit_fl)
        for idx in range(max_idx)
    ]

    if cat:
        bad_rates.sort(key=lambda x: x.bad_rate)

    overall_rate = None if cat else sum(b.bad for b in bad_rates) / sum(b.total for b in bad_rates)

    return BinBadRatesResult(
        bad_rates=bad_rates,
        overall_rate=overall_rate,
        bins=bins
    )


def _calc_max_bins(bins: List, max_bins_percentage: float) -> int: # Renamed max_bins to max_bins_percentage
    """Calculate maximum number of bins based on a percentage of current bins, with a minimum fallback."""
    if not isinstance(bins, list):
        # Log: print("Warning: 'bins' argument to _calc_max_bins was not a list. Returning MIN_BINS_FALLBACK.")
        return MIN_BINS_FALLBACK
    if not isinstance(max_bins_percentage, float) or not (0.0 < max_bins_percentage <= 1.0) :
        # Log: print(f"Warning: max_bins_percentage '{max_bins_percentage}' is not a float strictly between 0 and 1. Defaulting to MIN_BINS_FALLBACK.")
        return MIN_BINS_FALLBACK
    return max(int(len(bins) * max_bins_percentage), MIN_BINS_FALLBACK)


def prepare_data(data: pd.DataFrame,
                special_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare data for binning"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data should be pandas data frame")

    if special_cols:
        data = data.drop(special_cols, axis=1)

    feature_names = list(data.columns)
    return data, feature_names


def find_cat_features(x: pd.DataFrame,
                     feature_names: List[str],
                     cat_features_threshold: int) -> List[str]:
    """Find categorical features based on dtype and unique value count."""
    if not isinstance(x, pd.DataFrame):
        raise TypeError("Input 'x' must be a pandas DataFrame.")
    if not isinstance(feature_names, list):
        raise TypeError("Input 'feature_names' must be a list.")
    if not isinstance(cat_features_threshold, int) or cat_features_threshold < 0:
        raise ValueError("cat_features_threshold must be a non-negative integer.")

    if not feature_names or x.empty:
        return []

    categorical_found = []
    for feature_name in feature_names:
        if feature_name not in x.columns:
            # Log: print(f"Warning: Feature '{feature_name}' not found in DataFrame columns during categorical check.")
            continue

        col_dtype = x[feature_name].dtype
        is_object_dtype = np.issubdtype(col_dtype, np.object_)

        try:
            num_unique_values = len(pd.unique(x[feature_name].dropna().astype(str)))
        except Exception as e:
            # Log: print(f"Warning: Could not determine unique values for feature {feature_name}: {e}. Assuming not categorical by unique count.")
            num_unique_values = cat_features_threshold + 1

        if is_object_dtype or (num_unique_values < cat_features_threshold):
            categorical_found.append(feature_name)

    return categorical_found


def _cat_binning(x: np.ndarray, y: np.ndarray,
                 min_pct_group: float,
                 max_bins: Union[int, float],
                 diff_woe_threshold: float) -> Tuple[List[BadRates], Optional[str]]:
    """Perform categorical binning"""
    missing_bin = None
    x, data_type = _prepare_categorical_data(x)
    bins, bad_rates = _initialize_categorical_bins(x, y, max_bins)

    if len(y[pd.isna(x)]) > 0:
        x, bins, bad_rates, missing_bin = _handle_categorical_missing_values(x, y, bins, bad_rates, data_type)

    if len(bins) <= 2:
        return bad_rates, missing_bin

    bad_rates, bins = _enforce_diff_woe_categorical(x, y, bad_rates, bins, diff_woe_threshold)

    if len(bins) <= 2:
        return bad_rates, missing_bin

    # Resolve max_bins if it's a percentage
    if isinstance(max_bins, float) and max_bins < 1.0: # type: ignore
        # Assuming `bins` here refers to the current state of categorical bins (List[List[Union[str,float]]])
        # _calc_max_bins expects a list of bin boundaries or similar, not list of lists of values.
        # For categorical, len(bins) is the current number of bins.
        actual_max_bins = _calc_max_bins(bins, max_bins) # type: ignore
    elif isinstance(max_bins, int):
        actual_max_bins = max_bins
    else: # Fallback or error, for now, let's assume it resolves to a sensible integer.
          # This case might need more robust handling depending on expected inputs for max_bins.
        actual_max_bins = len(bins) # Default to current number of bins if type is unexpected.


    bad_rates, bins = _enforce_min_pct_group_categorical(x, y, bad_rates, bins, min_pct_group, actual_max_bins)

    return bad_rates, missing_bin


def _enforce_min_pct_group_categorical(x: np.ndarray, y: np.ndarray, bad_rates: List[BadRates], bins: List[List[Union[str, float]]], min_pct_group: float, max_bins_val: int) -> Tuple[List[BadRates], List[List[Union[str, float]]]]:
    """Ensures each categorical bin meets the minimum percentage group threshold and respects max_bins."""
    while (min(b.pct for b in bad_rates) <= min_pct_group or len(bad_rates) > max_bins_val) and len(bins) > 2:
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins, cat=True)
        # _merge_bins_min_pct for cat=True already updates bins to be List[List[Union[str,float]]]
        # from result.bad_rates where bin is List[Union[str,float]]
    return bad_rates, bins


def _enforce_diff_woe_categorical(x: np.ndarray, y: np.ndarray, bad_rates: List[BadRates], bins: List[List[Union[str, float]]], diff_woe_threshold: float) -> Tuple[List[BadRates], List[List[Union[str, float]]]]:
    """Ensures WOE differences between categorical bins meet the threshold."""
    while (idx := _check_diff_woe(tuple(bad_rates), diff_woe_threshold)) is not None and len(bad_rates) > 2:
        # Ensure idx is within bounds for bins modification
        if idx < len(bins) -1 : # Check if idx+1 is a valid index
            bins[idx + 1].extend(bins[idx])
            del bins[idx]
            result = _bin_bad_rates(x, y, bins, cat=True)
            bad_rates = result.bad_rates
            # Update bins from result to reflect any sorting or changes from _bin_bad_rates
            processed_bins = []
            for br in result.bad_rates:
                if isinstance(br.bin, list):
                    processed_bins.append(br.bin)
                else:
                    processed_bins.append([br.bin]) # Ensure bin items are lists
            bins = processed_bins
        else: # If idx is the last possible merge target, this logic might be flawed or an edge case
            break # Avoid index out of bounds
    return bad_rates, bins


def _prepare_categorical_data(x: np.ndarray) -> Tuple[np.ndarray, str]:
    """Prepares categorical data by converting to float or string and returns the data type."""
    try:
        x_prepared = x.astype(float)
        data_type = "float"
    except ValueError:
        x_prepared = x.astype(str)
        data_type = "object"
    return x_prepared, data_type


def _initialize_categorical_bins(x: np.ndarray, y: np.ndarray, max_bins: Union[int, float]) -> Tuple[List[List[Union[str, float]]], List[BadRates]]:
    """Initializes bins for categorical features."""
    unique_values = np.unique(x[~pd.isna(x)])
    bins = [[val] for val in unique_values]

    if max_bins < 1: # type: ignore
        max_bins = _calc_max_bins(bins, max_bins) # type: ignore

    if len(bins) > max_bins: # type: ignore
        # Calculate bad rates for each unique value
        bad_rates_dict: Dict[Union[str, float], float] = {
            b[0]: y[np.isin(x, b)].sum() / np.sum(np.isin(x, b)) if np.sum(np.isin(x, b)) > 0 else 0
            for b in bins
        }
        # Sort unique values by their bad rates
        sorted_bad_rates = dict(sorted(bad_rates_dict.items(), key=lambda item: item[1]))

        # Create quantiles based on sorted bad rates
        bad_rate_values = list(sorted_bad_rates.values())
        # Ensure max_bins is an int for range
        num_quantiles = int(max_bins) # type: ignore

        # Create quantile boundaries
        # Ensure q_list has at least two elements for zip to work
        if num_quantiles > 0:
            q_list = [0.0] + [
                np.nanquantile(bad_rate_values, q / num_quantiles) # Use num_quantiles which is int
                for q in range(1, num_quantiles) # Use num_quantiles
            ] + [1.0]
            q_list = sorted(list(set(q_list))) # Ensure unique and sorted
        else: # Fallback if num_quantiles is 0 (e.g. if max_bins was 0 or very small float)
            q_list = [0.0, 1.0]


        new_bins: List[List[Union[str, float]]] = []
        # Properly iterate through sorted_bad_rates items
        sorted_items = list(sorted_bad_rates.items())
        current_q_idx = 0

        # Handle cases with very few unique values or quantiles
        if not q_list or len(q_list) < 2: # Ensure q_list has at least two points to form a range
             if sorted_items: # If there are items, put them all in one bin
                 new_bins.append([item[0] for item in sorted_items])
        else:
            # More robust way to build new_bins based on quantiles
            current_bin_values: List[Union[str, float]] = []
            item_idx = 0
            for q_idx in range(len(q_list) -1):
                lower_bound = q_list[q_idx]
                upper_bound = q_list[q_idx+1]

                # Collect items for the current quantile bin
                temp_bin_values: List[Union[str, float]] = []
                while item_idx < len(sorted_items):
                    val, rate = sorted_items[item_idx]
                    # Special handling for the last bin to include all remaining items
                    if q_idx == len(q_list) - 2: # If this is the second to last q_list item, it defines the start of the last bin
                         if rate >= lower_bound:
                            temp_bin_values.append(val)
                         else: # Should not happen if sorted correctly
                            break
                    elif rate >= lower_bound and rate < upper_bound :
                        temp_bin_values.append(val)
                    elif rate >= upper_bound: # Value belongs to the next bin
                        break
                    item_idx +=1

                if temp_bin_values:
                    if new_bins and not current_bin_values: # If new_bins is not empty and current_bin is (first iteration)
                        new_bins[-1].extend(temp_bin_values)
                    elif current_bin_values: # if current_bin_values is not empty merge it
                         current_bin_values.extend(temp_bin_values)
                         new_bins.append(current_bin_values)
                         current_bin_values = []
                    else: # if both new_bins and current_bin_values are empty
                         new_bins.append(temp_bin_values)
                # Ensure last bin is added if it has values and wasn't processed
                if item_idx < len(sorted_items) and q_idx == len(q_list) -2 and not temp_bin_values:
                     remaining_values = [item[0] for item in sorted_items[item_idx:]]
                     if new_bins and not current_bin_values : # if new_bins is not empty and current_bin is (first iteration)
                         new_bins[-1].extend(remaining_values)
                     else:
                         new_bins.append(remaining_values)


        # If new_bins is empty (e.g., due to no unique values or issues in quantile logic),
        # fall back to original unique value bins or a single bin if no values.
        if not new_bins and unique_values.size > 0:
            # This might happen if q_list logic doesn't produce bins,
            # fallback to one bin per unique value if that's fewer than max_bins,
            # or a more robust quantile approach might be needed.
            # For now, let's ensure bins are not empty if unique_values exist.
             bins = [[val] for val in unique_values] # Re-initialize if new_bins is empty
        elif new_bins:
             bins = new_bins


        result = _bin_bad_rates(x, y, bins, cat=True)
        # bins = [b.bin for b in result.bad_rates] # This line might cause issues if b.bin is not always a list
        # Ensure bins are correctly structured for further processing
        processed_bins = []
        for b_rate in result.bad_rates:
            if isinstance(b_rate.bin, list):
                processed_bins.append(b_rate.bin)
            else: # If b_rate.bin is not a list, wrap it in a list
                processed_bins.append([b_rate.bin])
        bins = processed_bins
        bad_rates = result.bad_rates

    else: # len(bins) <= max_bins
        result = _bin_bad_rates(x, y, bins, cat=True)
        bad_rates = result.bad_rates
        # Ensure bins structure is consistent
        processed_bins = []
        for b_rate in result.bad_rates:
            if isinstance(b_rate.bin, list):
                processed_bins.append(b_rate.bin)
            else:
                processed_bins.append([b_rate.bin]) # Wrap if not a list
        bins = processed_bins


    return bins, bad_rates


def _handle_categorical_missing_values(x: np.ndarray, y: np.ndarray, bins: List[List[Union[str, float]]], bad_rates: List[BadRates], data_type: str) -> Tuple[np.ndarray, List[List[Union[str, float]]], List[BadRates], str]:
    """Handles missing values for categorical features."""
    missing_bin: Optional[str] = None
    missing_val = "Missing" if data_type == "object" else -1.0 # Use float for consistency if numeric

    if not bins: # Handle case where bins might be empty
        bins.append([missing_val]) # Create a new bin for missing values
        x[pd.isna(x)] = missing_val
        result = _bin_bad_rates(x, y, bins, cat=True) # Recalculate bad_rates
        bad_rates = result.bad_rates
        # Ensure bins are updated from result if _bin_bad_rates modifies them (e.g. sorts)
        # Extract and update bins from result.bad_rates to ensure consistency
        processed_bins = []
        for br in result.bad_rates:
            if isinstance(br.bin, list):
                processed_bins.append(br.bin)
            else:
                processed_bins.append([br.bin]) # Ensure bin items are lists
        bins = processed_bins
        missing_bin = "first" # Or some default as there's only one bin
        return x, bins, bad_rates, missing_bin # type: ignore

    if len(bins) < 2: # If only one bin exists (it might be the missing_val bin itself if all values were missing initially)
        # Check if the existing bin is the missing bin or if we need to add missing_val to it or a new one
        # This logic might need adjustment based on how _initialize_categorical_bins handles fully NaN series
        if not any(missing_val in sublist for sublist in bins): # If missing_val is not already in a bin
            bins[0].append(missing_val) # Add to the first bin if only one exists
        x[pd.isna(x)] = missing_val
        result = _bin_bad_rates(x, y, bins, cat=True)
        bad_rates = result.bad_rates
        # Update bins from result
        processed_bins = []
        for br in result.bad_rates:
            if isinstance(br.bin, list):
                processed_bins.append(br.bin)
            else:
                processed_bins.append([br.bin])
        bins = processed_bins
        # Determine missing_bin based on the (potentially single) bin's content
        if bins and bins[0] and bins[0][0] == missing_val: # Check first element of first bin
             missing_bin = "first"
        else: # Default or more complex logic might be needed
             missing_bin = "last" # Fallback
    else: # len(bins) >= 2
        na_bad_rate = y[pd.isna(x)].sum() / len(y[pd.isna(x)]) if len(y[pd.isna(x)]) > 0 else 0

        # Ensure bad_rates is not empty and elements have bad_rate attribute
        first_bin_bad_rate = bad_rates[0].bad_rate if bad_rates and hasattr(bad_rates[0], 'bad_rate') else 0
        last_bin_bad_rate = bad_rates[-1].bad_rate if bad_rates and hasattr(bad_rates[-1], 'bad_rate') else 0

        if abs(na_bad_rate - first_bin_bad_rate) < abs(na_bad_rate - last_bin_bad_rate):
            missing_bin = "first"
            bins[0].append(missing_val)
        else:
            missing_bin = "last"
            bins[-1].append(missing_val)
        x[pd.isna(x)] = missing_val
        result = _bin_bad_rates(x, y, bins, cat=True)
        bad_rates = result.bad_rates
        # Update bins from result
        processed_bins = []
        for br in result.bad_rates:
            if isinstance(br.bin, list):
                processed_bins.append(br.bin)
            else:
                processed_bins.append([br.bin])
        bins = processed_bins

    return x, bins, bad_rates, missing_bin # type: ignore


def cat_processing(x: pd.Series,
                  y: Union[np.ndarray, pd.Series],
                  min_pct_group: float,
                  max_bins: Union[int, float],
                  diff_woe_threshold: float) -> Dict:
    """Process categorical feature"""
    bad_rates, missing_bin = _cat_binning(
        x=x.values, # type: ignore
        y=y,
        min_pct_group=min_pct_group,
        max_bins=max_bins,
        diff_woe_threshold=diff_woe_threshold,
    )

    return {
        x.name: bad_rates,
        "missing_bin": missing_bin,
        "type_feature": "cat"
    }


def _num_binning(x: np.ndarray,
                 y: np.ndarray,
                 min_pct_group: float,
                 max_bins: Union[int, float],
                 diff_woe_threshold: float,
                 merge_type: str) -> Tuple[List[BadRates], Optional[str]]:
    """Perform numeric binning"""
    missing_bin = None
    non_na_x = x[~pd.isna(x)] # Added this line

    bins, bad_rates = _initialize_numerical_bins(x, y, max_bins)

    if len(y[pd.isna(x)]) > 0:
        x, bins, bad_rates, missing_bin = _handle_numerical_missing_values(x, y, bins, bad_rates, non_na_x)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    bad_rates, bins = _ensure_monotonicity(x, y, bad_rates, bins, merge_type)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    bad_rates, bins = _enforce_min_pct_group(x, y, bad_rates, bins, min_pct_group)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    bad_rates, bins = _enforce_diff_woe(x, y, bad_rates, bins, diff_woe_threshold)

    return bad_rates, missing_bin


def _enforce_diff_woe(x: np.ndarray, y: np.ndarray, bad_rates: List[BadRates], bins: List, diff_woe_threshold: float) -> Tuple[List[BadRates], List]:
    """Ensures that WOE differences between bins meet the threshold."""
    while (idx := _check_diff_woe(tuple(bad_rates), diff_woe_threshold)) is not None and len(bad_rates) > 2:
        del bins[idx + 1]
        result = _bin_bad_rates(x, y, bins)
        bad_rates = result.bad_rates
    return bad_rates, bins


def _initialize_numerical_bins(x: np.ndarray, y: np.ndarray, max_bins: Union[int, float]) -> Tuple[List, List[BadRates]]:
    """Initializes numerical bins based on unique values or quantiles."""
    if max_bins < 1:
        max_bins = _calc_max_bins(list(np.unique(x[~pd.isna(x)])), max_bins) # type: ignore

    non_na_x = x[~pd.isna(x)]
    unique_vals = np.unique(non_na_x)

    if len(unique_vals) > max_bins: # type: ignore
        bins = [np.NINF] + [
            np.nanquantile(x, q/max_bins) # type: ignore
            for q in range(1, int(max_bins)) # type: ignore
        ]
        bins = list(np.unique(bins))
        if len(bins) == 2 and len(unique_vals) > 1: # Check if unique_vals has at least 2 elements
            bins.append(unique_vals[1])
    else:
        bins = [np.NINF] + sorted(list(unique_vals)) # Convert unique_vals to list before sorting
    bins.append(np.inf)

    result = _bin_bad_rates(x, y, bins)
    bad_rates = result.bad_rates

    if pd.isna(bad_rates[0].bad_rate) and len(bad_rates) > 2:
        del bins[1]
        result = _bin_bad_rates(x, y, bins)
        bad_rates = result.bad_rates
    return bins, bad_rates


def _handle_numerical_missing_values(x: np.ndarray, y: np.ndarray, bins: List, bad_rates: List[BadRates], non_na_x: np.ndarray) -> Tuple[np.ndarray, List, List[BadRates], str]:
    """Handles missing values for numerical features."""
    missing_bin: Optional[str] = None
    na_bad_rate = y[pd.isna(x)].sum() / len(y[pd.isna(x)])

    if len(bad_rates) == 2: # If there are only two bins (excluding potential missing bin)
        # Determine if missing values are more similar to the lower or upper bin
        if na_bad_rate < bad_rates[1].bad_rate: # Compare with the second bin's bad rate
            x = np.nan_to_num(x, nan=np.amin(non_na_x) - 1) # Impute with a value smaller than min
            bins = [np.NINF, np.amin(non_na_x)] + bins[1:] # Adjust bins accordingly
            missing_bin = "first"
        else:
            x = np.nan_to_num(x, nan=np.amax(non_na_x) + 1) # Impute with a value larger than max
            bins = bins[:2] + [np.amax(non_na_x), np.inf] # Adjust bins accordingly
            missing_bin = "last"
    else:
        # For more than two bins, compare missing bad rate to average of first/second halves
        mid = len(bad_rates) // 2
        avg_first_half = np.mean([b.bad_rate for b in bad_rates[:mid] if pd.notna(b.bad_rate)]) # Ensure not to include NaN in mean
        avg_second_half = np.mean([b.bad_rate for b in bad_rates[mid:] if pd.notna(b.bad_rate)]) # Ensure not to include NaN in mean

        if abs(na_bad_rate - avg_first_half) < abs(na_bad_rate - avg_second_half):
            x = np.nan_to_num(x, nan=np.amin(non_na_x)) # Impute with min value
            missing_bin = "first"
        else:
            x = np.nan_to_num(x, nan=np.amax(non_na_x)) # Impute with max value
            missing_bin = "last"

    result = _bin_bad_rates(x, y, bins)
    bad_rates = result.bad_rates
    return x, bins, bad_rates, missing_bin # type: ignore


def _ensure_monotonicity(x: np.ndarray, y: np.ndarray, bad_rates: List[BadRates], bins: List, merge_type: str) -> Tuple[List[BadRates], List]:
    """Ensures monotonicity of bad rates by merging bins."""
    merge_func = _merge_bins_chi if merge_type == "chi2" else _merge_bins_iv
    while not _mono_flags(bad_rates) and len(bad_rates) > 2:
        bad_rates, bins = merge_func(x, y, bad_rates, bins)
    return bad_rates, bins


def _enforce_min_pct_group(x: np.ndarray, y: np.ndarray, bad_rates: List[BadRates], bins: List, min_pct_group: float) -> Tuple[List[BadRates], List]:
    """Ensures each bin meets the minimum percentage group threshold."""
    while min(b.pct for b in bad_rates) <= min_pct_group and len(bad_rates) > 2:
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins)
    return bad_rates, bins


def num_processing(x: pd.Series,
                  y: Union[np.ndarray, pd.Series],
                  min_pct_group: float,
                  max_bins: Union[int, float],
                  diff_woe_threshold: float,
                  merge_type: str) -> Dict:
    """Process numeric feature"""
    bad_rates, missing_bin = _num_binning(
        x=x.values, # type: ignore
        y=y,
        min_pct_group=min_pct_group,
        max_bins=max_bins,
        diff_woe_threshold=diff_woe_threshold,
        merge_type=merge_type,
    )

    return {
        x.name: bad_rates,
        "missing_bin": missing_bin,
        "type_feature": "num"
    }


def _refit_woe_dict(x: np.ndarray,
                    y: np.ndarray,
                    bins: List,
                    type_feature: str,
                    missing_bin: str) -> List[BadRates]:
    """Refit WOE dictionary"""
    cat = type_feature == "cat"

    if cat:
        na_value = (-1.0 if np.issubdtype(x.dtype, np.floating)
                   or np.issubdtype(x.dtype, np.integer) else "Missing")
        x[pd.isna(x)] = na_value
    elif missing_bin == "first":
        na_value = np.nanmin(x[~np.isnan(x)]) - 1 # Use np.nanmin to handle potential NaNs after imputation
        x[np.isnan(x)] = na_value
    elif missing_bin == "last":
        na_value = np.nanmax(x[~np.isnan(x)]) + 1 # Use np.nanmax to handle potential NaNs after imputation
        x[np.isnan(x)] = na_value

    result = _bin_bad_rates(x, y, bins, cat=cat, refit_fl=True)
    return result.bad_rates


def refit(x: pd.Series,
          y: np.ndarray,
          bins: List,
          type_feature: str,
          missing_bin: str) -> Dict:
    """Refit model"""
    bad_rates = _refit_woe_dict(
        x=x.values, # type: ignore
        y=y,
        bins=bins,
        type_feature=type_feature,
        missing_bin=missing_bin,
    )

    return {
        x.name: bad_rates,
        "missing_bin": missing_bin,
        "type_feature": type_feature
    }
