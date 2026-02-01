import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from woe_scoring.core.main import WOETransformer

class TestFeatureGeneration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 500
        self.data = pd.DataFrame({
            'A': np.random.rand(n) * 10,
            'B': np.random.rand(n) * 10,
            'C': np.random.choice(['X', 'Y', 'Z'], n)
        })
        # Target: related to Ratio A/B
        # If A/B > 1 -> good (1)
        self.data['ratio'] = self.data['A'] / (self.data['B'] + 0.01)
        self.target = (self.data['ratio'] > 1).astype(int)
        
        # Add some noise to make it realistic
        noise = np.random.binomial(1, 0.1, n)
        self.target = np.abs(self.target - noise)
        
        # Clean up manual feature
        del self.data['ratio']

    def test_generate_features(self):
        transformer = WOETransformer(
            max_bins=5,
            min_pct_group=0.05,
            generate_features=True,
            max_generated_features=10,
            n_jobs=1
        )
        
        transformer.fit(self.data, self.target)
        
        # Check if features generated
        generated_features = [f for f in transformer.feature_names if 'RATIO' in f or 'STATS' in f]
        self.assertTrue(len(generated_features) > 0, "No features generated")
        print(f"Generated features: {generated_features}")
        
        # Check if RATIO_A_DIV_B is likely selected (since it's predictive)
        self.assertTrue(any('RATIO' in f for f in generated_features))
        
        # Check transformed data
        transformed = transformer.transform(self.data)
        
        # Check columns
        for f in generated_features:
            self.assertIn(f"WOE_{f}", transformed.columns)
            
        # Test Save/Load
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            transformer.save_to_file(tmp_path)
            
            # Check file content
            with open(tmp_path, 'r') as f:
                content = json.load(f)
            
            # Check for metadata
            has_meta = False
            for item in content:
                if 'metadata' in item:
                    has_meta = True
                    break
            self.assertTrue(has_meta, "Metadata not found in saved file")
            
            # Load into new transformer
            new_transformer = WOETransformer()
            new_transformer.load_woe_iv_dict(tmp_path)
            
            # Transform with new transformer
            new_transformed = new_transformer.transform(self.data)
            
            # Compare results
            # Columns might be ordered differently if dict order varies, but WOE values should match
            # Sort columns to ensure comparison works
            transformed_sorted = transformed.reindex(sorted(transformed.columns), axis=1)
            new_transformed_sorted = new_transformed.reindex(sorted(new_transformed.columns), axis=1)
            
            pd.testing.assert_frame_equal(transformed_sorted, new_transformed_sorted)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_safe_original_data_with_generated_features(self):
        # Test safe_original_data=False (default)
        transformer = WOETransformer(
            max_bins=5,
            generate_features=True,
            safe_original_data=False,
            n_jobs=1
        )
        transformer.fit(self.data, self.target)
        transformed = transformer.transform(self.data)
        
        # Original columns and generated raw columns should be gone, only WOE_ columns remain
        for col in self.data.columns:
            self.assertNotIn(col, transformed.columns)
            
        generated_features = [f for f in transformer.feature_names if 'RATIO' in f or 'STATS' in f]
        for f in generated_features:
            self.assertNotIn(f, transformed.columns) # Raw generated feature should be dropped
            self.assertIn(f"WOE_{f}", transformed.columns) # WOE version should exist

        # Test safe_original_data=True
        transformer_safe = WOETransformer(
            max_bins=5,
            generate_features=True,
            safe_original_data=True,
            n_jobs=1
        )
        transformer_safe.fit(self.data, self.target)
        transformed_safe = transformer_safe.transform(self.data)
        
        # Original columns should be present
        for col in self.data.columns:
            self.assertIn(col, transformed_safe.columns)
            
        # Generated features might be present depending on implementation details of transform
        # The current implementation generates them into 'data', and if safe_original_data=True, 
        # it avoids dropping them.
        generated_features_safe = [f for f in transformer_safe.feature_names if 'RATIO' in f or 'STATS' in f]
        for f in generated_features_safe:
            self.assertIn(f, transformed_safe.columns) # Raw generated feature should be kept
            self.assertIn(f"WOE_{f}", transformed_safe.columns)

if __name__ == '__main__':
    unittest.main()
