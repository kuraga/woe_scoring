from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Any
# from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import chisquare
# from scipy.stats.contingency import chi2_contingency # Alternative if manual is complex
import logging # Add logging

logger = logging.getLogger(__name__) # Module-level logger

# Define a fallback value for max_bins if percentage calculation yields zero or invalid results
MIN_BINS_FALLBACK = 20 # Example value, can be adjusted


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
    overall_good: int = 0
    overall_bad: int = 0


# Define a fallback value for max_bins if percentage calculation yields zero or invalid results
MIN_BINS_FALLBACK = 20 # Example value, can be adjusted


def _chi2(bad_rates: List[BadRates], all_bad: int, all_good: int) -> float:
    """Calculate chi-square statistic for a list of bins (BadRates)."""
    # Chi-square test of independence on a 2xK contingency table (Good/Bad vs Bins)
    # Observed frequencies: [[good1, bad1], [good2, bad2], ...]
    # Expected frequencies: [[expected_good1, expected_bad1], [expected_good2, expected_bad2], ...]
    # Expected_bad_i = total_i * (all_bad / total_all)
    # Expected_good_i = total_i * (all_good / total_all)

    if all_bad == 0 and all_good == 0:
        # Cannot calculate chi2 if there's no data or target distribution
        return 0.0
    if len(bad_rates) <= 1:
         # Cannot calculate chi2 with 1 or zero bins
         return 0.0

    observed = []
    total_all = all_bad + all_good

    if total_all == 0: # Should be caught by first check, but double safe
        return 0.0

    for bin_rate in bad_rates:
        # Use original bad/total counts before smoothing for chi2
        observed_bad = bin_rate.bad
        observed_good = bin_rate.total - observed_bad # Assuming total is original total

        if observed_bad is None or bin_rate.total is None: # Defensive check
             continue

        observed.append([observed_good, observed_bad])

    if not observed:
        return 0.0

    try:
        observed_arr = np.array(observed, dtype=float)  # Use float to avoid integer division issues
        row_totals = observed_arr.sum(axis=1, keepdims=True) # Totals per bin
        col_totals = observed_arr.sum(axis=0, keepdims=True) # Overall good/bad totals
        grand_total = observed_arr.sum() # Overall total (should be sum(b.total))

        if grand_total == 0: # Should be same as total_all
            return 0.0

        expected_arr = (row_totals * col_totals) / grand_total

        # Calculate chi-square statistic manually for the contingency table
        # Flatten observed and expected arrays
        observed_flat = observed_arr.flatten()
        expected_flat = expected_arr.flatten()

        # Avoid division by zero for expected counts
        # Add a small epsilon to expected counts if they are zero
        expected_flat[expected_flat == 0] = 1e-9 # Small epsilon to avoid division by zero

        chi2_stat = np.sum((observed_flat - expected_flat)**2 / expected_flat)

        return float(chi2_stat)  # Ensure return is a simple float
    except Exception as e:
        logger.warning(f"Error in chi2 calculation: {e}")
        return 0.0  # Return 0 on error


# @lru_cache(maxsize=128)
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


def _find_index_of_diff_flag(bad_rates: List[BadRates]) -> Optional[int]:
    """Find index where monotonicity changes. Returns index or None if monotonic or <= 1 bin diff."""
    if len(bad_rates) <= 1: # Need at least 2 points to calculate diff
        return None
    bad_rate_diffs = np.diff([b.bad_rate for b in bad_rates])
    if len(bad_rate_diffs) == 0: # Should not happen if len(bad_rates) > 1, but defensive
         return None
    diffs_bool = bad_rate_diffs > 0
    # np.argmin(boolean array) finds the first index where the value is False.
    # If all diffs are > 0 (all True), np.argmin(True, True, ...) -> 0.
    # If all diffs are < 0 (all False), np.argmin(False, False, ...) -> 0.
    # This means np.argmin returns 0 if it's monotonic (all diffs same sign).
    # We only care about non-monotonic changes.
    # Check if all diffs have the same sign.
    first_diff_sign = np.sign(bad_rate_diffs[0]) if bad_rate_diffs[0] != 0 else 0
    is_monotonic = all(np.sign(d) == first_diff_sign or d == 0 for d in bad_rate_diffs)

    if is_monotonic:
        return None # It's monotonic

    # If not monotonic, find the first index where the sign changes relative to the previous one.
    # Or, the index where the bad_rate is 'out of line'.
    # The original logic np.argmin(diffs_bool) finds the first index where diffs_bool is False.
    # E.g., diffs = [0.1, -0.2, 0.3]. diffs_bool = [True, False, True]. argmin = 1. This seems correct.
    # E.g., diffs = [-0.1, 0.2, -0.3]. diffs_bool = [False, True, False]. argmin = 0. This also seems correct.
    return int(np.argmin(diffs_bool))


def _merge_bins_chi(x: np.ndarray, y: np.ndarray,
                   bad_rates: List[BadRates], bins: List) -> Tuple[List[BadRates], List]:
    """Merge bins using chi-square method"""
    # Ensure bad_rates has at least 2 items to proceed with merging
    if len(bad_rates) <= 1:
        # No merging possible or needed
        result = _bin_bad_rates(x, y, bins) # Re-calculate in case of initial smoothing
        return result.bad_rates, result.bins # Return original bins and re-calculated rates

    idx = _find_index_of_diff_flag(bad_rates)
    if idx is None: # Should be monotonic at this point if length > 1
         result = _bin_bad_rates(x, y, bins)
         return result.bad_rates, result.bins # Return original bins and re-calculated rates


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
    # The original _chi2 signature changed, need to pass all_bad and all_good.
    # These are calculated within _bin_bad_rates.
    # We need to calculate _bin_bad_rates for each potential merge scenario and then call _chi2.

    # Scenario 1: Merge bin at idx and idx+1 (delete bin boundary at idx+1)
    temp_bins_1 = bins.copy()
    # Ensure idx + 1 is a valid index before deleting
    if idx + 1 < len(temp_bins_1):
        del temp_bins_1[idx + 1]
        temp_result_1 = _bin_bad_rates(x, y, temp_bins_1)
        # Ensure result is valid before calculating chi2
        if temp_result_1 and temp_result_1.bad_rates:
             chi_1 = _chi2(temp_result_1.bad_rates, temp_result_1.overall_bad, temp_result_1.overall_good)
        else:
             chi_1 = np.inf # Treat as worst chi2 if binning failed or no rates
    else: # Cannot perform this merge scenario
        chi_1 = np.inf


    # Scenario 2: Merge bin at idx+1 and idx+2 (delete bin boundary at idx+2)
    temp_bins_2 = bins.copy()
    # Ensure idx + 2 is a valid index before deleting
    if idx + 2 < len(temp_bins_2):
         del temp_bins_2[idx + 2]
         temp_result_2 = _bin_bad_rates(x, y, temp_bins_2)
         # Ensure result is valid before calculating chi2
         if temp_result_2 and temp_result_2.bad_rates:
              chi_2 = _chi2(temp_result_2.bad_rates, temp_result_2.overall_bad, temp_result_2.overall_good)
         else:
              chi_2 = np.inf # Treat as worst chi2 if binning failed or no rates
    else: # Cannot perform this merge scenario
         chi_2 = np.inf


    # Compare chi-square values and delete the boundary that results in a lower chi-square
    # Lower chi-square indicates better grouping (more similar bad rates).
    if chi_1 == np.inf and chi_2 == np.inf:
        # Log warning: print("Warning: Neither chi-square merge scenario was possible or successful.")
        # No merging possible from this state. Do nothing or handle explicitly.
        pass # Keep original bins if neither merge worked
    elif chi_1 < chi_2:
        # Ensure idx + 1 is a valid index in the original bins before deleting
        if idx + 1 < len(bins):
            del bins[idx + 1]
    elif chi_2 < np.inf: # Only delete if chi_2 was calculated successfully and is better
         # Ensure idx + 2 is a valid index in the original bins before deleting
         if idx + 2 < len(bins):
              del bins[idx + 2]
    # If chi_1 == chi_2, default to deleting idx + 1 or do nothing if both inf. The current if/elif handles this.



def _merge_bins_iv(x: np.ndarray, y: np.ndarray,
                  bad_rates: List[BadRates], bins: List) -> Tuple[List[BadRates], List]:
    """Merge bins using IV method"""
    # Ensure bad_rates has at least 2 items to proceed with merging
    if len(bad_rates) <= 1:
        # No merging possible or needed
        result = _bin_bad_rates(x, y, bins) # Re-calculate in case of initial smoothing
        return result.bad_rates, result.bins # Return original bins and re-calculated rates

    idx = _find_index_of_diff_flag(bad_rates)
    if idx is None: # Should be monotonic at this point if length > 1
         result = _bin_bad_rates(x, y, bins)
         return result.bad_rates, result.bins # Return original bins and re-calculated rates


    # The original logic for idx == 0 or idx == len(bad_rates) - 2 assumes specific merges at ends.
    # Let's rely on _extract_bin_by_iv to handle boundary cases based on comparing IVs.
    _extract_bin_by_iv(bins, idx, x, y)

    # Re-calculate bad rates and bins after merging
    result = _bin_bad_rates(x, y, bins)
    # Note: _bin_bad_rates for numerical doesn't modify bins list structure, only content/order of BadRates.
    # So returning result.bins (which is the input bins list) is fine here.
    return result.bad_rates, result.bins


def _extract_bin_by_iv(bins: List, idx: int, x: np.ndarray, y: np.ndarray) -> None:
    """Extract bin based on IV values"""
    temp_bins = bins.copy()
    # Ensure idx + 1 is a valid index before attempting deletion
    if idx + 1 < len(temp_bins):
        del temp_bins[idx + 1]
        temp_result = _bin_bad_rates(x, y, temp_bins)
        iv_1 = sum(b.iv for b in temp_result.bad_rates) if temp_result and temp_result.bad_rates else -np.inf # Assign low IV if merge failed
    else:
        iv_1 = -np.inf # Cannot perform this merge

    temp_bins = bins.copy()
    # Ensure idx + 2 is a valid index before attempting deletion
    if idx + 2 < len(temp_bins):
        del temp_bins[idx + 2]
        temp_result = _bin_bad_rates(x, y, temp_bins)
        iv_2 = sum(b.iv for b in temp_result.bad_rates) if temp_result and temp_result.bad_rates else -np.inf # Assign low IV if merge failed
    else:
        iv_2 = -np.inf # Cannot perform this merge


    # Keep the merge that results in higher total IV
    if iv_1 == -np.inf and iv_2 == -np.inf:
         # Log warning: print("Warning: Neither IV merge scenario was possible or successful.")
         pass # Keep original bins
    elif iv_1 >= iv_2: # If IVs are equal, prefer the first merge
         # Ensure idx + 1 is valid before deleting from original bins
         if idx + 1 < len(bins):
              del bins[idx + 1]
    elif iv_2 > -np.inf: # Only delete if iv_2 was calculated successfully and is better
         # Ensure idx + 2 is valid before deleting from original bins
         if idx + 2 < len(bins):
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

    # For numerical, _bin_bad_rates does not reorder bins, so result.bins is just the input bins.
    # For categorical, _bin_bad_rates sorts bad_rates by bad_rate, and result.bins extracts
    # the bin lists in the new sorted order. We should use the sorted bins for subsequent steps.
    result = _bin_bad_rates(x, y, bins, cat=cat)
    # Ensure bins are updated with potentially sorted/restructured bins from result
    updated_bins_list = []
    for br in result.bad_rates:
        if isinstance(br.bin, list):
            updated_bins_list.append(br.bin)
        else:
             # This shouldn't happen for categorical bins typically, but as a fallback
             updated_bins_list.append([br.bin])
    bins = updated_bins_list

    return result.bad_rates, bins


@dataclass
class BinBadRatesResult:
    """Results container for bin bad rates calculation"""
    bad_rates: List[BadRates]
    overall_rate: Optional[float] = None
    bins: List = field(default_factory=list)
    overall_good: int = 0 # Added overall good count
    overall_bad: int = 0 # Added overall bad count


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
    # Ensure y is not empty before summing
    if len(y) == 0:
        # Log warning: print("Warning: Target variable y is empty in _bin_bad_rates.")
        return BinBadRatesResult(bad_rates=[], bins=bins, overall_good=0, overall_bad=0)

    all_bad_sum = int(y.sum())
    all_total_sum = len(y)
    all_good_sum = all_total_sum - all_bad_sum

    # Handle edge case: if all_bad_sum or all_good_sum is 0 for the entire dataset,
    # WOE/IV for all bins should be 0 as there's no separation ability.
    # _calc_stats already handles division by zero for term_good/term_bad,
    # but the overall IV will also be 0 if either all_bad_sum or all_good_sum is 0.
    # Let's just proceed; _calc_stats and IV summation will correctly result in 0.0.

    max_idx = len(bins) if cat or refit_fl else len(bins) - 1

    # Handle case where no bins are generated (e.g., empty or all NaN input)
    if max_idx <= 0:
         # Log warning: print("Warning: No bins generated in _bin_bad_rates.")
         return BinBadRatesResult(bad_rates=[], bins=bins, overall_good=all_good_sum, overall_bad=all_bad_sum)

    try:
        bad_rates = [
            _calc_stats(x, y, idx, all_bad_sum, all_good_sum, bins, cat, refit_fl)
            for idx in range(max_idx)
        ]
    except Exception as e:
        logger.warning(f"Error calculating stats in _bin_bad_rates: {e}")
        return BinBadRatesResult(bad_rates=[], bins=bins, overall_good=all_good_sum, overall_bad=all_bad_sum)

    # Remove any None results from _calc_stats if it returns None in edge cases (currently it doesn't)
    # bad_rates = [br for br in bad_rates if br is not None]


    if cat:
        # For categorical, sort bins by bad rate as part of standard procedure
        bad_rates.sort(key=lambda x: x.bad_rate)

    # Recalculate overall_rate based on the *final* bad_rates list content
    # This is useful if some bins were skipped or filtered, though the current _calc_stats shouldn't do that.
    total_sum_in_bad_rates = sum(br.total for br in bad_rates)
    bad_sum_in_bad_rates = sum(br.bad for br in bad_rates)
    overall_rate = bad_sum_in_bad_rates / total_sum_in_bad_rates if total_sum_in_bad_rates > 0 else 0.0

    return BinBadRatesResult(
        bad_rates=bad_rates,
        overall_rate=overall_rate,
        bins=bins, # Keep the input bins structure for numerical
        overall_good=all_good_sum, # Return total good counts
        overall_bad=all_bad_sum # Return total bad counts
    )


def _calc_max_bins(bins: List, max_bins_percentage: float) -> int: # Renamed max_bins to max_bins_percentage
    """Calculate maximum number of bins based on a percentage of current bins, with a minimum fallback."""
    # The 'bins' argument to this function, when called from _initialize_numerical_bins,
    # is `list(np.unique(x[~pd.isna(x)]))`. So it's a list of unique values.
    # When called from _initialize_categorical_bins, it's `bins` which is `[[val], [val], ...]`.
    # In both cases, len(bins) is the number of distinct initial bins or unique values.
    if not isinstance(bins, list):
        # Log: print("Warning: 'bins' argument to _calc_max_bins was not a list. Returning MIN_BINS_FALLBACK.")
        return MIN_BINS_FALLBACK
    if not isinstance(max_bins_percentage, float) or not (0.0 < max_bins_percentage <= 1.0) :
         # Log: print(f"Warning: max_bins_percentage '{max_bins_percentage}' is not a float strictly between 0 and 1. Defaulting to MIN_BINS_FALLBACK.")
        return MIN_BINS_FALLBACK

    # Calculate the target number of bins based on the percentage of current 'bins' count
    target_bins = int(len(bins) * max_bins_percentage)

    # Ensure the result is at least MIN_BINS_FALLBACK (e.g., 20)
    # and not more than the original number of bins if max_bins_percentage is near 1.0
    return max(target_bins, MIN_BINS_FALLBACK)


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
    missing_bin: Optional[str] = None # Initialize missing_bin to None
    x_prepared, data_type = _prepare_categorical_data(x)

    # Handle missing values BEFORE initial binning if they exist in the original data.
    # This ensures missing values are part of the data used for calculating initial bad rates and quantiles.
    # The missing values will be imputed/represented by a placeholder.
    if pd.isna(x).any(): # Check if there are *any* missing values in original x
        x_prepared, missing_val = _impute_categorical_missing_values(x_prepared, data_type) # Impute, get placeholder

    # Proceed with binning on the data where NaNs have been replaced by a placeholder
    bins, bad_rates = _initialize_categorical_bins(x_prepared, y, max_bins)

    # Determine the missing_bin strategy based on the placeholder bin
    if pd.isna(x).any(): # Only determine missing_bin if there were missing values
        missing_bin = _determine_missing_bin_strategy(bins, bad_rates, x_prepared, y)


    if len(bins) <= 1: # If after initialization and potentially missing handling, we have 1 or 0 bins
        # If there's only one bin, it's likely the missing bin or the only unique value.
        # Return whatever rates were calculated for this single bin.
        return bad_rates, missing_bin


    # Resolve max_bins if it's a percentage AFTER initial binning based on unique values
    if isinstance(max_bins, float) and max_bins < 1.0: # type: ignore
        # `bins` is now List[List[Union[str, float]]]
        actual_max_bins = _calc_max_bins(bins, max_bins) # type: ignore
    elif isinstance(max_bins, int):
        actual_max_bins = max_bins
    else: # Fallback, should not happen with current validation
        actual_max_bins = len(bins) # Default to current number of bins

    # Ensure `bad_rates` is updated based on current `bins` before applying merging logic
    # (This is handled within the enforce functions by calling _bin_bad_rates)

    # Enforce difference WOE threshold merging
    # _enforce_diff_woe_categorical updates bad_rates and bins
    bad_rates, bins = _enforce_diff_woe_categorical(x_prepared, y, bad_rates, bins, diff_woe_threshold)

    if len(bins) <= 1: # Check again after WOE merging
        return bad_rates, missing_bin

    # Enforce minimum percentage group threshold and max_bins
    # _enforce_min_pct_group_categorical updates bad_rates and bins
    bad_rates, bins = _enforce_min_pct_group_categorical(x_prepared, y, bad_rates, bins, min_pct_group, actual_max_bins)


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
    return x_prepared, data_type


def _prepare_categorical_data(x: np.ndarray) -> Tuple[np.ndarray, str]:
    """Prepares categorical data by converting to float or string and returns the data type."""
    try:
        x_prepared = x.astype(float)
        data_type = "float"
    except ValueError:
        x_prepared = x.astype(str)
        data_type = "object"
    return x_prepared, data_type


def _impute_categorical_missing_values(x: np.ndarray, data_type: str) -> Tuple[np.ndarray, Union[str, float]]:
    """Imputes missing values for categorical features with a placeholder."""
    missing_val = "Missing" if data_type == "object" else -1.0 # Use float for consistency if numeric
    # Create a copy to avoid modifying the original array outside this function's scope
    x_imputed = x.copy()
    x_imputed[pd.isna(x_imputed)] = missing_val
    return x_imputed, missing_val


def _determine_missing_bin_strategy(bins: List[List[Union[str, float]]], bad_rates: List[BadRates], x: np.ndarray, y: np.ndarray) -> Optional[str]:
    """
    Determines the missing_bin strategy ('first', 'last', or None) after binning.
    Assumes missing values were imputed with a placeholder *before* initial binning.
    Checks where the placeholder value ended up in the sorted bins.
    """
    # Determine the placeholder value that was used
    # This is a bit tricky without knowing the placeholder directly.
    # We can try to infer it based on the data type that was used for imputation.
    # Assuming -1.0 for numeric-like categories and "Missing" for object.
    # This logic needs to be consistent with _impute_categorical_missing_values.
    inferred_data_type = "object" if np.issubdtype(x.dtype, np.object_) else "float" # Check dtype of imputed x
    placeholder_val = "Missing" if inferred_data_type == "object" else -1.0

    # Check if the placeholder value is present in any of the bins
    placeholder_bin_index = -1
    for i, bin_list in enumerate(bins):
        if placeholder_val in bin_list:
            placeholder_bin_index = i
            break

    if placeholder_bin_index == -1:
        # Log warning: print(f"Warning: Missing value placeholder {placeholder_val} not found in any bin after binning.")
        return None # Missing bin not identified in the bins

    # Now, determine if this bin is the first or last among the *sorted* bad_rates.
    # The `bins` list should correspond to the order in `bad_rates` if `_bin_bad_rates` sorts them.
    if placeholder_bin_index == 0:
        return "first"
    elif placeholder_bin_index == len(bins) - 1:
        return "last"
    else:
        # Log warning: print(f"Warning: Missing value placeholder bin is not the first or last bin (index {placeholder_bin_index}/{len(bins)-1}). Cannot determine simple strategy.")
        return None # Cannot assign a simple first/last strategy if missing bin is in the middle


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

    # Convert BadRates objects to dictionaries for serialization
    # Handle case when bad_rates is empty
    if not bad_rates:
        bad_rates_dicts = []
    else:
        # Convert each BadRates object to a dictionary with all fields
        bad_rates_dicts = []
        for br in bad_rates:
            bad_rates_dict = {
                "bin": br.bin,
                "total": br.total,
                "bad": br.bad,
                "pct": br.pct,
                "bad_rate": br.bad_rate,
                "woe": br.woe,
                "iv": br.iv
            }
            bad_rates_dicts.append(bad_rates_dict)

    return {
        x.name: bad_rates_dicts,
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

    # Make a copy of the input to avoid modifying the original
    x_copy = x.copy()
    non_na_x = x_copy[~pd.isna(x_copy)]

    # Check if there's enough non-missing data
    if len(non_na_x) == 0:
        # All values are missing, create a single bin for missing values
        bins = [np.NINF, np.inf]
        result = _bin_bad_rates(x_copy, y, bins)
        return result.bad_rates, "first"  # All values go to the first bin

    # Initialize bins and bad rates
    bins, bad_rates = _initialize_numerical_bins(x_copy, y, max_bins)

    # Handle missing values if present
    if np.isnan(x_copy).any():
        x_copy, bins, bad_rates, missing_bin = _handle_numerical_missing_values(x_copy, y, bins, bad_rates, non_na_x)

    # Skip further processing if we have 2 or fewer bins
    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    # Ensure monotonicity of bad rates
    bad_rates, bins = _ensure_monotonicity(x_copy, y, bad_rates, bins, merge_type)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    # Enforce minimum percentage group
    bad_rates, bins = _enforce_min_pct_group(x_copy, y, bad_rates, bins, min_pct_group)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    # Enforce difference in WOE values
    bad_rates, bins = _enforce_diff_woe(x_copy, y, bad_rates, bins, diff_woe_threshold)

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
    # Handle case where all values are NaN
    non_na_x = x[~pd.isna(x)]
    if len(non_na_x) == 0:
        bins = [np.NINF, np.inf]  # Create a single bin for all values
        result = _bin_bad_rates(x, y, bins)
        return bins, result.bad_rates

    if isinstance(max_bins, float) and max_bins < 1:
        max_bins = _calc_max_bins(list(np.unique(non_na_x)), max_bins) # type: ignore

    unique_vals = np.unique(non_na_x)

    # Create initial bins based on unique values or quantiles
    if len(unique_vals) > max_bins: # type: ignore
        try:
            bins = [np.NINF] + [
                np.quantile(non_na_x, q/max_bins) # Use quantile instead of nanquantile for non-NaN data
                for q in range(1, int(max_bins)) # type: ignore
            ]
            bins = list(np.unique(bins))
            if len(bins) == 2 and len(unique_vals) > 1:
                bins.append(unique_vals[1])
        except Exception as e:
            logger.warning(f"Error creating quantile bins: {e}")
            bins = [np.NINF, np.inf]  # Fallback to a single bin
    else:
        bins = [np.NINF] + sorted(list(unique_vals)) # Convert unique_vals to list before sorting

    bins.append(np.inf)

    result = _bin_bad_rates(x, y, bins)
    bad_rates = result.bad_rates

    # Handle NaN bad_rates in the first bin
    if bad_rates and len(bad_rates) > 1 and (pd.isna(bad_rates[0].bad_rate) or bad_rates[0].bad_rate is None) and len(bins) > 2:
        del bins[1] # Merge the first bin with the second
        result = _bin_bad_rates(x, y, bins)
        bad_rates = result.bad_rates

    return bins, bad_rates


def _handle_numerical_missing_values(x: np.ndarray, y: np.ndarray, bins: List, bad_rates: List[BadRates], non_na_x: np.ndarray) -> Tuple[np.ndarray, List, List[BadRates], str]:
    """Handles missing values for numerical features."""
    missing_bin = None

    # Check if there are any missing values to handle
    if not pd.isna(x).any() or len(y[pd.isna(x)]) == 0:
        # No missing values to handle
        return x, bins, bad_rates, missing_bin

    # Calculate bad rate for missing values
    na_bad_rate = y[pd.isna(x)].sum() / len(y[pd.isna(x)])

    # If no non-NaN values, handle as a special case
    if len(non_na_x) == 0:
        x_copy = x.copy()
        x_copy[pd.isna(x)] = -999999.0  # Use a placeholder value
        missing_bin = "first"
        result = _bin_bad_rates(x_copy, y, bins)
        return x_copy, bins, result.bad_rates, missing_bin

    if len(bad_rates) == 2: # If there are only two bins (excluding potential missing bin)
        # Determine if missing values are more similar to the lower or upper bin
        if na_bad_rate < bad_rates[1].bad_rate: # Compare with the second bin's bad rate
            x_copy = x.copy()
            x_copy = np.nan_to_num(x_copy, nan=np.amin(non_na_x) - 1) # Impute with a value smaller than min
            bins = [np.NINF, np.amin(non_na_x)] + bins[1:] # Adjust bins accordingly
            missing_bin = "first"
        else:
            x_copy = x.copy()
            x_copy = np.nan_to_num(x_copy, nan=np.amax(non_na_x) + 1) # Impute with a value larger than max
            bins = bins[:2] + [np.amax(non_na_x), np.inf] # Adjust bins accordingly
            missing_bin = "last"
    else:
        # For more than two bins, compare missing bad rate to average of first/second halves
        mid = len(bad_rates) // 2
        avg_first_half = np.mean([b.bad_rate for b in bad_rates[:mid] if pd.notna(b.bad_rate)]) # Ensure not to include NaN in mean
        avg_second_half = np.mean([b.bad_rate for b in bad_rates[mid:] if pd.notna(b.bad_rate)]) # Ensure not to include NaN in mean

        x_copy = x.copy()
        if abs(na_bad_rate - avg_first_half) < abs(na_bad_rate - avg_second_half):
            x_copy = np.nan_to_num(x_copy, nan=np.amin(non_na_x)) # Impute with min value
            missing_bin = "first"
        else:
            x_copy = np.nan_to_num(x_copy, nan=np.amax(non_na_x)) # Impute with max value
            missing_bin = "last"

    result = _bin_bad_rates(x_copy, y, bins)
    bad_rates = result.bad_rates
    return x_copy, bins, bad_rates, missing_bin


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


def _impute_numerical_missing_values(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Union[float, str]]:
    """Imputes numerical missing values and returns the imputed value."""
    missing_val = np.nan # Start with original NaNs
    x_imputed = x.copy() # Work on a copy

    na_mask = pd.isna(x)
    if not na_mask.any():
        # Log info: print("No missing values to impute.")
        return x_imputed, np.nan # Return copy and NaN as no imputation happened

    non_na_x = x[~na_mask]

    if non_na_x.size == 0:
        # All values are NaN
        x_imputed[:] = -999999.0  # Use a large negative number as placeholder
        return x_imputed, -999999.0

    # For regular cases with some non-NaN values
    return x_imputed, np.nan


def num_processing(x: pd.Series,
                  y: Union[np.ndarray, pd.Series],
                  min_pct_group: float,
                  max_bins: Union[int, float],
                  diff_woe_threshold: float,
                  merge_type: str) -> Dict:
    """Process numeric feature"""
    # Pass the original series with potential NaNs to _num_binning
    bad_rates, missing_bin = _num_binning(
        x=x.values, # type: ignore
        y=y,
        min_pct_group=min_pct_group,
        max_bins=max_bins,
        diff_woe_threshold=diff_woe_threshold,
        merge_type=merge_type,
    )

    # Convert BadRates objects to dictionaries for serialization
    # Handle case when bad_rates is empty
    if not bad_rates:
        bad_rates_dicts = []
    else:
        # Convert each BadRates object to a dictionary with all fields
        bad_rates_dicts = []
        for br in bad_rates:
            bad_rates_dict = {
                "bin": br.bin,
                "total": br.total,
                "bad": br.bad,
                "pct": br.pct,
                "bad_rate": br.bad_rate,
                "woe": br.woe,
                "iv": br.iv
            }
            bad_rates_dicts.append(bad_rates_dict)

    return {
        x.name: bad_rates_dicts,
        "missing_bin": missing_bin,
        "type_feature": "num"
    }


def _refit_woe_dict(x: np.ndarray,
                    y: np.ndarray,
                    bins: List,
                    type_feature: str,
                    missing_bin: Optional[str]) -> List[BadRates]:
    """Refit WOE dictionary"""
    cat = type_feature == "cat"
    x_copy = x.copy() # Refit should work on a copy of the input array

    # Handle missing values based on the saved missing_bin strategy from the original fit
    # This assumes NaNs were handled during the original binning, and this strategy
    # dictates where original NaNs *would* have been grouped or how they were imputed.
    # For refit, we need to re-apply that logic or simply preserve the mapping for NaNs.
    # A simpler approach for refit might be to identify NaNs and map them directly to the WOE
    # of the bin designated by missing_bin, if that WOE is stored somewhere.
    # However, the `refit` function in `WOETransformer` recalculates bad rates for the *existing* bins.
    # So, we just need to make sure NaNs in the *refit data* `x` fall into a bin.
    # The simplest way is to impute NaNs in `x_copy` using values that will place them
    # into the designated 'missing_bin' equivalent.

    if pd.isna(x_copy).any():
        na_mask = pd.isna(x_copy)
        non_na_x_refit = x_copy[~na_mask]

        if cat:
            na_value = (-1.0 if np.issubdtype(x_copy.dtype, np.floating)
                       or np.issubdtype(x_copy.dtype, np.integer) else "Missing")
            # In categorical refit, the 'missing' category (e.g. -1.0 or 'Missing') is expected
            # to be one of the values within the saved `bins` list for that feature.
            # So we just need to impute NaNs with this placeholder.
            x_copy[na_mask] = na_value
        elif missing_bin == "first":
            # Impute NaNs with a value smaller than the minimum in the refit non-NA data
            # to ensure they fall into the first bin boundary during refit.
            # Handle empty non_na_x_refit case (all NaNs in refit data)
            min_val = np.nanmin(non_na_x_refit) if non_na_x_refit.size > 0 and np.isfinite(np.nanmin(non_na_x_refit)) else -1e9 # Use a large neg number if non_na_x is empty or all inf/-inf
            na_value = min_val - 1 # Use a value less than min
            x_copy[na_mask] = na_value
        elif missing_bin == "last":
             # Impute NaNs with a value larger than the maximum in the refit non-NA data
             # to ensure they fall into the last bin boundary during refit.
             # Handle empty non_na_x_refit case
            max_val = np.nanmax(non_na_x_refit) if non_na_x_refit.size > 0 and np.isfinite(np.nanmax(non_na_x_refit)) else 1e9 # Use a large pos number
            na_value = max_val + 1 # Use a value greater than max
            x_copy[na_mask] = na_value
        # If missing_bin is None for numerical, NaNs might not have been handled explicitly in fit,
        # or the strategy wasn't recorded. In that case, we might just impute with median/mean or keep NaNs
        # if the binning logic handles NaNs outside specific bins.
        # Given the current num_processing imputes NaNs, let's ensure consistency.
        # If missing_bin is None, this implies no specific strategy was recorded from fit.
        # The safest approach for refit is to impute with a value that doesn't disrupt existing bins,
        # like median/mean, or impute based on overall bad rate (closer to first/last).
        # Let's stick to the recorded missing_bin if available. If None, imputation might not be desired here.
        # Or maybe imputation to median/mean is a default if no missing_bin?
        # For now, only impute if missing_bin is 'first' or 'last'. If None, assume NaNs were not specially handled in fit.

    # Now re-calculate bad rates using the potentially imputed x_copy and the fixed bins.
    # For refit, _bin_bad_rates calculates stats for each bin based on the new x_copy and y.
    # The `bins` argument here is the list of bin boundaries/categories from the original fit.
    result = _bin_bad_rates(x_copy, y, bins, cat=cat, refit_fl=True)
    return result.bad_rates


def refit(x: pd.Series,
          y: np.ndarray,
          bins: List,
          type_feature: str,
          missing_bin: Optional[str]) -> Dict:
    """Refit model"""
    bad_rates = _refit_woe_dict(
        x=x.values, # type: ignore
        y=y,
        bins=bins,
        type_feature=type_feature,
        missing_bin=missing_bin,
    )

    # Convert BadRates objects to dictionaries for serialization
    # Handle case when bad_rates is empty
    if not bad_rates:
        bad_rates_dicts = []
    else:
        # Convert each BadRates object to a dictionary with all fields
        bad_rates_dicts = []
        for br in bad_rates:
            bad_rates_dict = {
                "bin": br.bin,
                "total": br.total,
                "bad": br.bad,
                "pct": br.pct,
                "bad_rate": br.bad_rate,
                "woe": br.woe,
                "iv": br.iv
            }
            bad_rates_dicts.append(bad_rates_dict)

    return {
        x.name: bad_rates_dicts,
        "missing_bin": missing_bin,
        "type_feature": type_feature
    }
