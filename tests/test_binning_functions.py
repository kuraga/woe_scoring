import unittest
from typing import Dict, List

import numpy as np
import pandas as pd

from woe_scoring.core.binning.functions import (
    _bin_event_rates,
    _calc_max_bins,
    _calc_stats,
    _cat_binning,
    _extract_bin_by_chi2,
    _extract_bin_by_iv,
    _merge_bins_chi,
    _merge_bins_iv,
    _merge_bins_min_pct,
    _num_binning,
    _refit_woe_dict,
    cat_processing,
    find_cat_features,
    num_processing,
    prepare_data,
    refit,
)


class TestBinningFunctions(unittest.TestCase):
    def setUp(self):
        # Create test data for binning
        np.random.seed(42)
        n_samples = 100

        # Create numeric features with different distributions
        self.num_data = pd.DataFrame(
            {
                "normal": np.random.normal(0, 1, n_samples),
                "uniform": np.random.uniform(-5, 5, n_samples),
                "skewed": np.exp(np.random.normal(0, 1, n_samples)),
            }
        )

        # Create categorical features
        self.cat_data = pd.DataFrame(
            {
                "low_card": np.random.choice(["A", "B", "C"], n_samples),
                "high_card": np.random.choice(["X", "Y", "Z", "W", "V"], n_samples),
                "imbalanced": np.random.choice(
                    ["Rare", "Common"], n_samples, p=[0.1, 0.9]
                ),
            }
        )

        # Create a target variable with binary outcome (0 or 1)
        self.y = np.random.binomial(1, 0.3, n_samples)

        # Create a combined dataset
        self.X = pd.concat([self.num_data, self.cat_data], axis=1)

        # Test input bins for some functions
        self.test_bins: List[Dict] = [
            {"event": 5, "total": 20, "event_rate": 0.25, "woe": 0.5},
            {"event": 10, "total": 20, "event_rate": 0.5, "woe": 0.0},
            {"event": 15, "total": 20, "event_rate": 0.75, "woe": -0.5},
            {"event": 18, "total": 20, "event_rate": 0.9, "woe": -1.0},
        ]

        # Calculate overall rate for tests
        total_event = sum(b["event"] for b in self.test_bins)
        total_count = sum(b["total"] for b in self.test_bins)
        self.overall_rate = total_event / total_count

    def test_prepare_data(self):
        """Test the prepare_data function"""
        # Test with numeric data
        X_prepared, feature_names = prepare_data(self.num_data, None)
        self.assertIsInstance(X_prepared, pd.DataFrame)
        # feature_names is now a list rather than Index
        self.assertEqual(len(feature_names), X_prepared.shape[1])

        # Test with categorical data
        X_prepared, feature_names = prepare_data(self.cat_data, None)
        self.assertIsInstance(X_prepared, pd.DataFrame)
        self.assertEqual(len(feature_names), X_prepared.shape[1])

        # Test with mixed data
        X_prepared, feature_names = prepare_data(self.X, None)
        self.assertIsInstance(X_prepared, pd.DataFrame)
        self.assertEqual(len(feature_names), X_prepared.shape[1])

    def test_find_cat_features(self):
        """Test the find_cat_features function"""
        # For testing purposes, create a simplified dataset with clear categorical columns
        test_data = pd.DataFrame(
            {
                "categorical1": ["A", "B", "C", "A", "B"],
                "categorical2": ["X", "Y", "X", "Y", "Z"],
                "numeric1": [1.1, 2.2, 3.3, 4.4, 5.5],
                "numeric2": [10, 20, 30, 40, 50],
            }
        )

        feature_names = test_data.columns.tolist()
        # Set threshold appropriately for our test data
        cat_features = find_cat_features(
            x=test_data, feature_names=feature_names, cat_features_threshold=5
        )

        # Check if categorical features are correctly identified
        self.assertIn("categorical1", cat_features)
        self.assertIn("categorical2", cat_features)

        # Check if numeric features are not included
        self.assertNotIn("numeric1", cat_features)
        self.assertNotIn("numeric2", cat_features)

    def test_calc_stats(self):
        """Test the _calc_stats function"""
        # Create a simple test dataframe
        test_df = pd.DataFrame(
            {"feature": ["A", "B", "A", "C", "B"], "target": [1, 0, 1, 0, 1]}
        )

        # Calculate the required parameters
        all_event = test_df["target"].sum()  # Sum of all event cases
        all_not_event = len(test_df["target"]) - all_event  # Total not_event cases
        bins = ["A", "B", "C"]  # List of unique values

        # Call _calc_stats with correct parameters
        result = _calc_stats(
            test_df["feature"].values,
            test_df["target"].values,
            0,
            all_event,
            all_not_event,
            bins,
            True,
        )

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("bin", result)
        self.assertIn("total", result)
        self.assertIn("event", result)
        self.assertIn("event_rate", result)
        self.assertIn("pct", result)

        # Check that the bin value is what we expect
        self.assertEqual(result["bin"], "A")

    def test_bin_event_rates(self):
        """Test the _bin_event_rates function"""
        # Create test arrays directly
        x = np.array([1, 2, 1, 3, 2])
        y = np.array([1, 0, 1, 0, 1])

        # Prepare the inputs
        # For categorical processing, bins must be a list of lists of values
        bins = [[1], [2], [3]]

        # Call _bin_event_rates with correct parameters
        result, _ = _bin_event_rates(x, y, bins, cat=True)

        # Check result structure
        self.assertIsInstance(result, list)
        for bin_dict in result:
            self.assertIn("bin", bin_dict)
            self.assertIn("total", bin_dict)
            self.assertIn("event", bin_dict)
            self.assertIn("event_rate", bin_dict)

        # Check calculations for specific bins
        # Note: result elements have 'bin' key which is the list [val]
        bin1 = next((b for b in result if b["bin"] == [1]), None)
        self.assertIsNotNone(bin1)
        self.assertEqual(bin1["total"], 2)
        self.assertEqual(bin1["event"], 2)
        self.assertEqual(bin1["event_rate"], 1.0)

        bin2 = next((b for b in result if b["bin"] == [2]), None)
        self.assertIsNotNone(bin2)
        self.assertEqual(bin2["total"], 2)
        self.assertEqual(bin2["event"], 1)
        self.assertEqual(bin2["event_rate"], 0.5)

    def test_calc_max_bins(self):
        """Test the _calc_max_bins function"""
        # Test with different sample sizes
        self.assertEqual(_calc_max_bins(10, 0.05), 2)  # min_pcnt = 0.05
        self.assertEqual(_calc_max_bins(10, 0.1), 2)  # min_pcnt = 0.1
        self.assertEqual(_calc_max_bins(20, 0.5), 10)  # max bins

    def test_merge_bins_chi(self):
        """Test the _merge_bins_chi function"""
        # Test data
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 0, 1, 0])

        # Calculate event rates
        event_rates = [
            {"bin": 1, "event": 1, "total": 1, "event_rate": 1.0},
            {"bin": 2, "event": 0, "total": 1, "event_rate": 0.0},
            {"bin": 3, "event": 1, "total": 1, "event_rate": 1.0},
            {"bin": 4, "event": 0, "total": 1, "event_rate": 0.0},
        ]

        bins = [1, 2, 3, 4]

        event_rates_result, bins_result = _merge_bins_chi(
            event_rates, bins, self.overall_rate
        )

        # Check that the result is a list with one less bin
        self.assertIsInstance(event_rates_result, list)
        self.assertEqual(len(event_rates_result), len(event_rates) - 1)

    def test_extract_bin_by_chi2(self):
        """Test the _extract_bin_by_chi2 function"""
        # Just test that _extract_bin_by_chi2 modifies the bins directly
        bins = [1, 2, 3, 4]
        idx = 1  # Example index

        # Create a copy for reference
        bins_copy = bins.copy()

        # Call the function - it operates on bins in-place
        _extract_bin_by_chi2(bins, idx, None, None)

        # Verify that bins has been modified
        self.assertNotEqual(bins, bins_copy)
        self.assertLess(len(bins), len(bins_copy))

        # We've already verified that bins has been modified above

    def test_merge_bins_iv(self):
        """Test the _merge_bins_iv function"""
        # Calculate event rates
        event_rates = [
            {"bin": 1, "event": 1, "total": 1, "event_rate": 1.0, "woe": 0.5},
            {"bin": 2, "event": 0, "total": 1, "event_rate": 0.0, "woe": 0.0},
            {"bin": 3, "event": 1, "total": 1, "event_rate": 1.0, "woe": -0.5},
            {"bin": 4, "event": 0, "total": 1, "event_rate": 0.0, "woe": -1.0},
        ]

        bins = [1, 2, 3, 4]

        # Test bin merging directly with the event_rates and bins
        event_rates_result, bins_result = _merge_bins_iv(event_rates, bins)

        # Check that the result is a list with one less bin
        self.assertIsInstance(event_rates_result, list)
        self.assertEqual(len(event_rates_result), len(event_rates) - 1)

    def test_extract_bin_by_iv(self):
        """Test the _extract_bin_by_iv function"""
        bins = [1, 2, 3, 4]
        idx = 1  # Example index

        # Make a copy to verify changes
        bins_copy = bins.copy()

        # _extract_bin_by_iv doesn't return a value, it modifies the bins in-place
        _extract_bin_by_iv(bins, idx, None, None)

        # Check that bins has been modified (should be shorter)
        self.assertIsInstance(bins, list)
        self.assertLess(len(bins), 4)  # Original length was 4

    def test_merge_bins_min_pct(self):
        """Test the _merge_bins_min_pct function"""
        # Calculate event rates
        event_rates = [
            {"bin": 1, "event": 1, "total": 5, "event_rate": 0.2, "pct": 0.02},
            {"bin": 2, "event": 2, "total": 10, "event_rate": 0.2, "pct": 0.04},
            {"bin": 3, "event": 5, "total": 100, "event_rate": 0.05, "pct": 0.4},
            {"bin": 4, "event": 2, "total": 135, "event_rate": 0.01, "pct": 0.54},
        ]

        bins = [1, 2, 3, 4]

        min_pcnt = 0.05
        event_rates_result, bins_result = _merge_bins_min_pct(
            event_rates, bins, min_pcnt
        )

        # Check that small bins are merged
        self.assertIsInstance(event_rates_result, list)
        self.assertLess(len(event_rates_result), len(event_rates))

        # Verify no bin has pct < min_pcnt
        for bin_dict in event_rates_result:
            self.assertGreaterEqual(bin_dict.get("pct", 0), min_pcnt)

    def test_cat_processing(self):
        """Test the cat_processing function"""
        # Test with categorical data
        feature = "low_card"

        result = cat_processing(
            x=self.cat_data[feature],
            y=self.y,
            min_pct_group=0.05,
            max_bins=10,
            diff_woe_threshold=0.05,
        )

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("low_card", result)
        self.assertIn("missing_bin", result)
        self.assertIn("type_feature", result)

        # Check the type_feature is set correctly
        self.assertEqual(result["type_feature"], "cat")

        # Check if result contains bin information
        self.assertIsInstance(result["low_card"], list)
        for bin_info in result["low_card"]:
            self.assertIn("bin", bin_info)
            self.assertIn("total", bin_info)
            self.assertIn("event", bin_info)
            self.assertIn("event_rate", bin_info)

    def test_num_processing(self):
        """Test the num_processing function"""
        # Test with numeric data
        feature = "normal"

        result = num_processing(
            x=self.num_data[feature],
            y=self.y,
            min_pct_group=0.05,
            max_bins=10,
            diff_woe_threshold=0.05,
            merge_type="chi2",
        )

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("normal", result)
        self.assertIn("missing_bin", result)
        self.assertIn("type_feature", result)

        # Check the type_feature is set correctly
        self.assertEqual(result["type_feature"], "num")

        # Check if result contains bin information
        self.assertIsInstance(result["normal"], list)
        for bin_info in result["normal"]:
            self.assertIn("bin", bin_info)
            self.assertIn("total", bin_info)
            self.assertIn("event", bin_info)
            self.assertIn("event_rate", bin_info)

    def test_refit_woe_dict(self):
        """Test the _refit_woe_dict function"""
        # Create bins structure (list of lists for categorical)
        bins = [["A"], ["B"], ["C"]]

        # Create test data with matching categories
        test_data = pd.Series(
            ["A", "B", "A", "C", "B", "D"]
        )  # 'D' is a new category (missing)
        test_target = pd.Series([1, 0, 1, 0, 1, 1])

        missing_bin = "first"

        # Call with correct signature
        new_woe_list = _refit_woe_dict(
            x=test_data.values,
            y=test_target.values,
            bins=bins,
            type_feature="cat",
            missing_bin=missing_bin,
        )

        # Check that we get back a list of bin stats
        self.assertIsInstance(new_woe_list, list)
        self.assertGreater(len(new_woe_list), 0)

        # Check structure of the first bin result
        first_bin = new_woe_list[0]
        self.assertIsInstance(first_bin, dict)
        self.assertIn("woe", first_bin)
        self.assertIn("bin", first_bin)

        # Test with NaN values
        test_data_nan = pd.Series(["A", "B", np.nan])
        test_target_nan = pd.Series([1, 0, 1])

        new_woe_list_nan = _refit_woe_dict(
            x=test_data_nan.values,
            y=test_target_nan.values,
            bins=bins,
            type_feature="cat",
            missing_bin="first",
        )
        # Verify result is valid
        self.assertIsInstance(new_woe_list_nan, list)


if __name__ == "__main__":
    unittest.main()
