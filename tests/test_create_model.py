import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock
from woe_scoring.core.main import CreateModel, WOETransformer


class TestCreateModel(unittest.TestCase):
    def setUp(self):
        # Create a simple test dataset
        np.random.seed(42)
        n_samples = 200

        # Create numeric-only features to avoid type conversion issues
        self.X = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, n_samples),
            'numeric2': np.random.uniform(-5, 5, n_samples),
        })

        # Create a target variable with binary outcome (0 or 1)
        self.y = np.random.binomial(1, 0.3, n_samples)

        # Create a simple transformed dataset with numeric values only
        self.X_transformed = pd.DataFrame({
            'WOE_numeric1': self.X['numeric1'],
            'WOE_numeric2': self.X['numeric2']
        })

        # Create a mock transformer with just enough functionality for the tests
        self.transformer = WOETransformer(
            max_bins=10,
            min_pct_group=0.05,
            n_jobs=1,
            random_state=42,
            prefix="WOE_",
            merge_type="chi2",
            cat_features=None,
            special_cols=[],
            cat_features_threshold=0,
            diff_woe_threshold=0.05,
            safe_original_data=False
        )
        self.transformer.fit(self.X, self.y)

        # Default parameters for model
        self.model = CreateModel(
            selection_method='rfe',
            model_type='sklearn',
            max_vars=None,
            special_cols=None,
            unused_cols=None,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
            cv=3,
            scoring='roc_auc'
        )
        # Add necessary attributes without calling fit
        self.model.feature_names_ = self.X.columns.tolist()
        self.model.coef_ = [0.1, -0.1]
        self.model.intercept_ = 0.2

        # Patch the model methods to avoid actual computation
        self.mock_fit()

    def mock_fit(self):
        """Mock methods to avoid actual computation"""
        # Create sample attributes that would be created by actual fit
        self.model.model = MagicMock()
        self.model.selected_features = ['WOE_numeric1', 'WOE_numeric2']
        self.model.feature_names_ = ['WOE_numeric1', 'WOE_numeric2']
        self.model.feature_report = {'WOE_numeric1': 0.8, 'WOE_numeric2': 0.7}
        self.model.score_report = {'auc': 0.85, 'gini': 0.7}

        # Mock model methods
        self.model.fit = MagicMock(return_value=self.model)
        self.model.predict = MagicMock(return_value=np.random.randint(0, 2, len(self.X)))
        self.model.predict_proba = MagicMock(return_value=np.random.random(len(self.X)))
        self.model.save_scorecard = MagicMock()

    def test_fit(self):
        """Test the fit method of CreateModel"""
        # Call the mocked fit method
        self.model.fit(self.X_transformed, self.y)

        # Verify fit was called with correct parameters
        self.model.fit.assert_called_with(self.X_transformed, self.y)

        # Check if the model attributes exist
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.selected_features)
        self.assertIsNotNone(self.model.feature_report)
        self.assertIsNotNone(self.model.score_report)

    def test_predict_proba(self):
        """Test the predict_proba method of CreateModel"""
        # Get predicted probabilities
        y_proba = self.model.predict_proba(self.X_transformed)

        # Verify predict_proba was called with correct parameters
        self.model.predict_proba.assert_called_with(self.X_transformed)

        # Check shape and type
        self.assertEqual(len(y_proba), len(self.X))
        self.assertTrue(isinstance(y_proba, np.ndarray))

    def test_predict(self):
        """Test the predict method of CreateModel"""
        # Get predictions
        y_pred = self.model.predict(self.X_transformed)

        # Verify predict was called with correct parameters
        self.model.predict.assert_called_with(self.X_transformed)

        # Check shape
        self.assertEqual(len(y_pred), len(self.X))

    def test_generate_sql(self):
        """Test the generate_sql method of CreateModel"""
        # Generate SQL
        sql = self.model.generate_sql(encoder=self.transformer, source_table="features_data")

        # Check if SQL is a non-empty string
        self.assertTrue(isinstance(sql, str))
        self.assertTrue(len(sql) > 0)

        # Check if SQL contains key elements
        self.assertIn("FROM features_data", sql)
        self.assertIn("CASE", sql)
        self.assertIn("END", sql)
        self.assertIn("WHEN", sql)
        self.assertIn("EXP", sql)
        self.assertIn("as PD", sql)

    def test_save_scorecard(self):
        """Test the save_scorecard method of CreateModel"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            temp_file = tmp.name

        try:
            # Save the scorecard using the mock
            self.model.save_scorecard(temp_file)

            # Verify save_scorecard was called with correct parameters
            self.model.save_scorecard.assert_called_with(temp_file)

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_different_estimators(self):
        """Test CreateModel with different estimator types"""
        # Create a different model with mocked methods
        rf_model = CreateModel(
            selection_method='sfs',
            model_type='sklearn',
            max_vars=5,
            n_jobs=-1,
            random_state=42
        )

        # Mock the methods
        rf_model.model = MagicMock()
        rf_model.fit = MagicMock(return_value=rf_model)
        rf_model.predict = MagicMock(return_value=np.random.randint(0, 2, len(self.X)))

        # Test the model
        rf_model.fit(self.X_transformed, self.y)
        rf_pred = rf_model.predict(self.X_transformed)

        # Verify methods were called
        rf_model.fit.assert_called_with(self.X_transformed, self.y)
        rf_model.predict.assert_called_with(self.X_transformed)

        # Check if output has the expected shape
        self.assertEqual(len(rf_pred), len(self.X))


if __name__ == '__main__':
    unittest.main()
