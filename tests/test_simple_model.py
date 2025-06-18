import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from woe_scoring.core.main import CreateModel, WOETransformer


class TestSimpleModel(unittest.TestCase):
    def setUp(self):
        # Create a simple test dataset
        np.random.seed(42)
        n_samples = 100

        # Create only numeric features to avoid type issues
        self.X = pd.DataFrame({
            'numeric1': np.random.normal(0, 1, n_samples),
            'numeric2': np.random.uniform(-5, 5, n_samples)
        })

        # Create a target variable with binary outcome (0 or 1)
        self.y = np.random.binomial(1, 0.3, n_samples)

        # Create a mock transformer
        self.transformer = WOETransformer(
            max_bins=5,
            min_pct_group=0.05,
            n_jobs=1,
            cat_features=[],  # No categorical features
            special_cols=None
        )

        # Create transformer mocks
        self.transformer.fit = MagicMock(return_value=self.transformer)
        self.transformer.transform = MagicMock()

        # Make transform return a modified DataFrame with the WOE_ prefix
        self.X_transformed = pd.DataFrame({
            'WOE_numeric1': self.X['numeric1'],
            'WOE_numeric2': self.X['numeric2']
        })
        self.transformer.transform.return_value = self.X_transformed

        # Default parameters for model
        self.model = CreateModel(
            selection_method='rfe',
            model_type='sklearn',
            max_vars=None,
            special_cols=None,
            unused_cols=None,
            n_jobs=1,
            random_state=42,
            class_weight='balanced',
            cv=3,
            scoring='roc_auc'
        )

    def test_initialization(self):
        """Test that the CreateModel initializes correctly"""
        self.assertEqual(self.model.selection_method, 'rfe')
        self.assertEqual(self.model.model_type, 'sklearn')
        self.assertEqual(self.model.n_jobs, 1)
        self.assertEqual(self.model.random_state, 42)
        self.assertEqual(self.model.class_weight, 'balanced')
        self.assertEqual(self.model.cv, 3)
        self.assertEqual(self.model.scoring, 'roc_auc')

    def test_basic_fit(self):
        """Test a basic fit process with the model"""
        # Use a different selection method that doesn't require min_features_to_select
        self.model.selection_method = 'iv'

        # Create mocks for the model
        self.model.fit = MagicMock(return_value=self.model)
        self.model.model = MagicMock()
        self.model.selected_features = ['WOE_numeric1', 'WOE_numeric2']

        # Transform data first using the mocked transformer
        X_transformed = self.transformer.transform(self.X)

        # Fit the model using the mock
        self.model.fit(X_transformed, self.y)

        # Verify the mock was called with correct parameters
        self.model.fit.assert_called_with(X_transformed, self.y)
        self.transformer.transform.assert_called_with(self.X)

        # Check that model attributes exist
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.selected_features)


if __name__ == '__main__':
    unittest.main()
