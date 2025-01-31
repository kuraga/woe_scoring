import pytest
import numpy as np
import pandas as pd
from woe_scoring.core.binning.functions import (_chi2, _mono_flags, _merge_bins_chi,
                                              _merge_bins_iv, _merge_bins_min_pct)

def test_chi2_calculation():
    from woe_scoring.core.binning.functions import BadRates

    bad_rates = [
        BadRates(bad=10, total=100, pct=0.5, bad_rate=0.1, woe=0.0, iv=0.1, bin=[0, 1]),
        BadRates(bad=20, total=100, pct=0.5, bad_rate=0.2, woe=0.2, iv=0.2, bin=[1, 2])
    ]

    chi2_value = _chi2(bad_rates, overall_rate=0.15)
    assert isinstance(chi2_value, float)
    assert chi2_value >= 0

def test_monotonic_check():
    from woe_scoring.core.binning.functions import BadRates

    # Monotonic increasing bad rates
    monotonic_rates = [
        BadRates(bad=10, total=100, pct=0.5, bad_rate=0.1, woe=0.0, iv=0.1, bin=[0, 1]),
        BadRates(bad=20, total=100, pct=0.5, bad_rate=0.2, woe=0.2, iv=0.2, bin=[1, 2]),
        BadRates(bad=30, total=100, pct=0.5, bad_rate=0.3, woe=0.3, iv=0.3, bin=[2, 3])
    ]

    assert _mono_flags(monotonic_rates) == True

    # Non-monotonic bad rates
    non_monotonic_rates = [
        BadRates(bad=10, total=100, pct=0.5, bad_rate=0.1, woe=0.0, iv=0.1, bin=[0, 1]),
        BadRates(bad=30, total=100, pct=0.5, bad_rate=0.3, woe=0.3, iv=0.3, bin=[1, 2]),
        BadRates(bad=20, total=100, pct=0.5, bad_rate=0.2, woe=0.2, iv=0.2, bin=[2, 3])
    ]

    assert _mono_flags(non_monotonic_rates) == False

def test_merge_bins_functions(sample_data):
    df, y = sample_data
    x = df['numeric_feature'].values

    # Create initial bins
    bins = [float('-inf'), -1, 0, 1, float('inf')]

    # Test chi-square merging
    bad_rates, new_bins = _merge_bins_chi(x, y, [], bins)
    assert len(new_bins) < len(bins)

    # Test IV merging
    bad_rates, new_bins = _merge_bins_iv(x, y, [], bins)
    assert len(new_bins) < len(bins)

    # Test minimum percentage merging
    bad_rates, new_bins = _merge_bins_min_pct(x, y, [], bins)
    assert len(new_bins) < len(bins)
