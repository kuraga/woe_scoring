import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000

    # Create numeric feature
    x_num = np.random.normal(0, 1, n_samples)

    # Create categorical feature
    categories = ['A', 'B', 'C']
    x_cat = np.random.choice(categories, n_samples)

    # Create target variable with some relationship to features
    y = (x_num > 0).astype(int) | (x_cat == 'A').astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'numeric_feature': x_num,
        'categorical_feature': x_cat
    })

    return df, pd.Series(y)

@pytest.fixture
def woe_transformer():
    from woe_scoring import WOETransformer
    return WOETransformer(
        max_bins=5,
        min_pct_group=0.05,
        n_jobs=1,
        cat_features=['categorical_feature']
    )
