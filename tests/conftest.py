import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def sample_data():
    df = pd.DataFrame({
        'numeric_feature': np.arange(40), # More predictable numeric data
        'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], size=40)
    })
    y = pd.Series(np.random.randint(0, 2, size=40))
    return df, y

@pytest.fixture(scope="module")
def simple_data_for_woe():
    """Creates a simple dataset for predictable WOE calculation."""
    data = pd.DataFrame({
        'num_feat': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'cat_feat': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'D']
    })
    # Target: Make 'A' and 'C' mostly 0, 'B' and 'D' mostly 1 for distinct WOEs
    # Num_feat: low values mostly 0, high values mostly 1
    target = pd.Series(
        [0,0,0,0,1,  # num 1-5 (A)
         0,1,1,1,1,  # num 6-10 (B)
         0,0,0,1,1,  # num 11-15 (C)
         0,1,1,1,1]  # num 16-20 (D)
    )
    return data, target

@pytest.fixture(scope="module")
def fitted_woe_transformer(simple_data_for_woe):
    """A WOETransformer fitted on simple_data_for_woe."""
    from woe_scoring import WOETransformer # Import locally for fixture
    data, target = simple_data_for_woe
    transformer = WOETransformer(
        cat_features=['cat_feat'],
        max_bins=2, # For num_feat, to get fewer, more predictable bins
        min_pct_group=0.05,
        diff_woe_threshold=0.01,
        prefix="WOE_"
    )
    transformer.fit(data, target)
    return transformer
