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

    if not good or not bad:
        good += 0.5
        bad += 0.5

    woe = np.log((good / all_good) / (bad / all_bad))
    iv = ((good / all_good) - (bad / all_bad)) * woe

    return BadRates(
        bin=value,
        total=total,
        bad=bad,
        pct=pct,
        bad_rate=bad_rate,
        woe=woe,
        iv=iv
    )


def _bin_bad_rates(x: np.ndarray, y: np.ndarray, bins: List,
                   cat: bool = False, refit_fl: bool = False) -> BinBadRatesResult:
    """Calculate bad rates for all bins"""
    all_bad = int(y.sum())
    all_good = len(y) - all_bad
    max_idx = len(bins) if cat or refit_fl else len(bins) - 1

    bad_rates = [
        _calc_stats(x, y, idx, all_bad, all_good, bins, cat, refit_fl)
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

    try:
        x = x.astype(float)
        data_type = "float"
    except ValueError:
        x = x.astype(str)
        data_type = "object"

    bins = [[_bin] for _bin in np.unique(x[~pd.isna(x)])]

    if max_bins < 1:
        max_bins = _calc_max_bins(bins, max_bins)

    if len(bins) > max_bins:
        bad_rates_dict = {
            bins[i][0]: y[np.isin(x, bins[i])].sum() / len(y[np.isin(x, bins[i])])
            for i in range(len(bins))
        }
        bad_rates_dict = dict(sorted(bad_rates_dict.items(), key=lambda item: item[1]))

        bad_rate_list = list(bad_rates_dict.values())
        q_list = [0.0] + [
            np.nanquantile(bad_rate_list, q/max_bins)
            for q in range(1, int(max_bins))
        ] + [1.0]
        q_list = sorted(set(q_list))

        new_bins = [[list(bad_rates_dict.keys())[0]]]
        start = 1
        for q1, q2 in zip(q_list[:-1], q_list[1:]):
            curr_bin = []
            for key, rate in list(bad_rates_dict.items())[start:]:
                if rate >= q2:
                    break
                if q1 <= rate < q2:
                    curr_bin.append(key)
                    start += 1
            if curr_bin:
                new_bins.append(curr_bin)

        result = _bin_bad_rates(x, y, new_bins, cat=True)
        bins = [b.bin for b in result.bad_rates]
        bad_rates = result.bad_rates
    else:
        result = _bin_bad_rates(x, y, bins, cat=True)
        bad_rates = result.bad_rates

    if len(y[pd.isna(x)]) > 0:
        missing_val = "Missing" if data_type == "object" else -1
        if len(bins) < 2:
            bins.append([])
            bins[1].append(missing_val)
            x[pd.isna(x)] = missing_val
            result = _bin_bad_rates(x, y, bins, cat=True)
            bad_rates = result.bad_rates
            missing_bin = "first" if bad_rates[0].bin[0] in ["Missing", -1] else "last"
        else:
            na_bad_rate = y[pd.isna(x)].sum() / len(y[pd.isna(x)])
            if abs(na_bad_rate - bad_rates[0].bad_rate) < abs(na_bad_rate - bad_rates[-1].bad_rate):
                missing_bin = "first"
                bins[0].append(missing_val)
            else:
                missing_bin = "last"
                bins[-1].append(missing_val)
            x[pd.isna(x)] = missing_val
            result = _bin_bad_rates(x, y, bins, cat=True)
            bad_rates = result.bad_rates

    if len(bins) <= 2:
        return bad_rates, missing_bin

    while (idx := _check_diff_woe(tuple(bad_rates), diff_woe_threshold)) is not None and len(bad_rates) > 2:
        bins[idx + 1].extend(bins[idx])
        del bins[idx]
        result = _bin_bad_rates(x, y, bins, cat=True)
        bad_rates = result.bad_rates

    if len(bins) <= 2:
        return bad_rates, missing_bin

    while (min(b.pct for b in bad_rates) <= min_pct_group or len(bad_rates) > max_bins) and len(bins) > 2:
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins, cat=True)

    return bad_rates, missing_bin


def cat_processing(x: pd.Series,
                  y: Union[np.ndarray, pd.Series],
                  min_pct_group: float,
                  max_bins: Union[int, float],
                  diff_woe_threshold: float) -> Dict:
    """Process categorical feature"""
    bad_rates, missing_bin = _cat_binning(
        x=x.values,
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

    if max_bins < 1:
        max_bins = _calc_max_bins(list(np.unique(x[~pd.isna(x)])), max_bins)

    non_na_x = x[~pd.isna(x)]
    unique_vals = np.unique(non_na_x)

    if len(unique_vals) > max_bins:
        bins = [np.NINF] + [
            np.nanquantile(x, q/max_bins)
            for q in range(1, int(max_bins))
        ]
        bins = list(np.unique(bins))
        if len(bins) == 2:
            bins.append(unique_vals[1])
    else:
        bins = [np.NINF] + sorted(unique_vals)
    bins.append(np.inf)

    result = _bin_bad_rates(x, y, bins)
    bad_rates = result.bad_rates

    if pd.isna(bad_rates[0].bad_rate) and len(bad_rates) > 2:
        del bins[1]
        result = _bin_bad_rates(x, y, bins)
        bad_rates = result.bad_rates

    if len(y[pd.isna(x)]) > 0:
        na_bad_rate = y[pd.isna(x)].sum() / len(y[pd.isna(x)])

        if len(bad_rates) == 2:
            if na_bad_rate < bad_rates[1].bad_rate:
                x = np.nan_to_num(x, nan=np.amin(non_na_x) - 1)
                bins = [np.NINF, np.amin(non_na_x)] + bins[1:]
                missing_bin = "first"
            else:
                x = np.nan_to_num(x, nan=np.amax(non_na_x) + 1)
                bins = bins[:2] + [np.amax(non_na_x), np.inf]
                missing_bin = "last"
        else:
            mid = len(bad_rates) // 2
            avg_first_half = np.mean([b.bad_rate for b in bad_rates[:mid]])
            avg_second_half = np.mean([b.bad_rate for b in bad_rates[mid:]])

            if abs(na_bad_rate - avg_first_half) < abs(na_bad_rate - avg_second_half):
                x = np.nan_to_num(x, nan=np.amin(non_na_x))
                missing_bin = "first"
            else:
                x = np.nan_to_num(x, nan=np.amax(non_na_x))
                missing_bin = "last"

        result = _bin_bad_rates(x, y, bins)
        bad_rates = result.bad_rates

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    merge_func = _merge_bins_chi if merge_type == "chi2" else _merge_bins_iv

    while not _mono_flags(bad_rates) and len(bad_rates) > 2:
        bad_rates, bins = merge_func(x, y, bad_rates, bins)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    while min(b.pct for b in bad_rates) <= min_pct_group and len(bad_rates) > 2:
        bad_rates, bins = _merge_bins_min_pct(x, y, bad_rates, bins)

    if len(bad_rates) <= 2:
        return bad_rates, missing_bin

    while (idx := _check_diff_woe(tuple(bad_rates), diff_woe_threshold)) is not None and len(bad_rates) > 2:
        del bins[idx + 1]
        result = _bin_bad_rates(x, y, bins)
        bad_rates = result.bad_rates

    return bad_rates, missing_bin


def num_processing(x: pd.Series,
                  y: Union[np.ndarray, pd.Series],
                  min_pct_group: float,
                  max_bins: Union[int, float],
                  diff_woe_threshold: float,
                  merge_type: str) -> Dict:
    """Process numeric feature"""
    bad_rates, missing_bin = _num_binning(
        x=x.values,
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
        na_value = np.nanmin(x[~np.isnan(x)]) - 1
        x[np.isnan(x)] = na_value
    elif missing_bin == "last":
        na_value = np.nanmax(x[~np.isnan(x)]) + 1
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
        x=x.values,
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
