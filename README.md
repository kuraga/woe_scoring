[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# WOE-Scoring

Monotone Weight Of Evidence (WOE) Transformer and LogisticRegression model with scikit-learn API. Optimized for performance and stability.

## Features

- **WOE Transformation**: Convert categorical and numerical features to Weight of Evidence encoding
- **Automated Feature Selection**: Multiple algorithms for optimal feature selection
- **Binning Strategies**: Smart binning with monotonicity constraints
- **Sklearn Compatibility**: Follows scikit-learn's API standards
- **Performance Optimized**: Parallel processing and vectorized operations
- **SQL Export**: Generate SQL for model deployment
- **Scorecard Generation**: Create credit scorecards with customizable scaling

## Installation

```bash
pip install woe-scoring
```

# Quickstart

1. Install the package:
```bash
pip install woe-scoring
```

2. Use WOETransformer:
```python
import pandas as pd
from woe_scoring import WOETransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("titanic_data.csv")
train, test = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["Survived"]
)

special_cols = [
    "PassengerId",
    "Survived",
    "Name",
    "Ticket",
    "Cabin",
]

cat_cols = [
    "Pclass",
    "Sex",
    "SibSp",
    "Parch",
    "Embarked",
]

encoder = WOETransformer(
    max_bins=8,
    min_pct_group=0.1,
    diff_woe_threshold=0.1,
    cat_features=cat_cols,
    special_cols=special_cols,
    n_jobs=-1,
    merge_type="chi2",
)

encoder.fit(train, train["Survived"])
encoder.save_to_file("train_dict.json")

encoder.load_woe_iv_dict("train_dict.json")
encoder.refit(train, train["Survived"])

enc_train = encoder.transform(train)
enc_test = encoder.transform(test)

model = LogisticRegression()
model.fit(enc_train, train["Survived"])
test_proba = model.predict_proba(enc_test)[:, 1]
```
3. Use CreateModel:

```python
import pandas as pd
from woe_scoring import CreateModel
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic_data.csv")
train, test = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["Survived"]
)

special_cols = [
    "PassengerId",
    "Survived",
    "Name",
    "Ticket",
    "Cabin",
]

model = CreateModel(
    max_vars=5,
    special_cols=special_cols,
    selection_method="sfs",
    model_type="sklearn",
    gini_threshold=5.0,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    cv=3,
)
model.fit(train, train["Survived"])
test_proba = model.predict_proba(test[model.feature_names_])

print(model.coef_, model.intercept_)
print(model.feature_names_)
```

## Detailed Documentation

### WOETransformer

The `WOETransformer` converts categorical and numerical features into Weight of Evidence (WOE) values. WOE measures the predictive power of a feature by comparing the distribution of events and non-events.

```python
WOETransformer(
    max_bins=10,               # Maximum number of bins for each feature
    min_pct_group=0.05,        # Minimum percentage of each bin
    n_jobs=1,                  # Number of parallel jobs
    prefix="WOE_",             # Prefix for transformed features
    merge_type="chi2",         # Bin merging strategy ('chi2', 'woe', 'monotonic')
    cat_features=None,         # List of categorical features
    special_cols=None,         # Columns to exclude from transformation
    cat_features_threshold=0,  # Threshold for auto-identifying categorical features
    diff_woe_threshold=0.05,   # Minimum WOE difference between bins
    safe_original_data=False   # Whether to keep original features
)
```

#### Key Methods

- `fit(data, target)`: Calculates optimal bins and WOE values
- `transform(data)`: Converts features to WOE values
- `save_to_file(path)`: Saves binning information to a JSON file
- `load_woe_iv_dict(path)`: Loads binning information from a JSON file
- `refit(data, target)`: Updates WOE values for existing bins with new data

### CreateModel

The `CreateModel` class combines feature selection, model training, and model evaluation:

```python
CreateModel(
    selection_method='rfe',    # Feature selection method ('rfe', 'sfs', 'iv')
    model_type='sklearn',      # Model implementation ('sklearn', 'statsmodel')
    max_vars=None,             # Maximum number of features to select
    special_cols=None,         # Columns to include as-is
    unused_cols=None,          # Columns to exclude
    n_jobs=1,                  # Number of parallel jobs
    gini_threshold=5.0,        # Minimum Gini score to keep a feature
    iv_threshold=0.05,         # Minimum IV threshold for feature selection
    corr_threshold=0.5,        # Correlation threshold for feature selection
    min_pct_group=0.05,        # Minimum percentage for each group
    random_state=None,         # Random seed for reproducibility
    class_weight='balanced',   # Class weighting strategy
    direction='forward',       # Direction for sequential feature selection
    cv=3,                      # Cross-validation folds
    l1_exp_scale=4,            # Exponent scale for L1 regularization
    l1_grid_size=20,           # Grid size for L1 regularization search
    scoring='roc_auc'          # Performance metric
)
```

#### Key Methods

- `fit(data, target)`: Selects features and fits model
- `predict(data)`: Makes binary predictions
- `predict_proba(data)`: Returns probability predictions
- `save_reports(path)`: Saves model reports
- `generate_sql(encoder)`: Generates SQL for model deployment
- `save_scorecard(encoder, path, ...)`: Creates credit scorecard

## Advanced Usage

### Generating SQL for Deployment

```python
# First fit the WOE transformer and model
encoder = WOETransformer()
encoder.fit(train, train["target"])
train_woe = encoder.transform(train)

model = CreateModel()
model.fit(train_woe, train["target"])

# Generate SQL query for scoring
sql_query = model.generate_sql(encoder)
```

### Creating a Scorecard

```python
# Save a credit scorecard to Excel
model.save_scorecard(
    encoder=encoder,
    path="output_dir",
    base_scorecard_points=600,  # Base score
    odds=50,                    # Base odds
    points_to_double_odds=20    # Points to double the odds
)
```

### Customizing Binning for Categorical Features

```python
# Specify categorical features and their treatment
encoder = WOETransformer(
    cat_features=["education", "marital_status", "occupation"],
    max_bins=5,                 # Max bins for categorical features
    diff_woe_threshold=0.1,     # Merge bins with similar WOE values
    min_pct_group=0.05          # Minimum population percentage per bin
)
```

## Performance Optimization

The library is optimized for performance with:
- Vectorized operations for fast transformation
- Parallel processing for binning and feature selection
- Efficient memory usage for large datasets
- Optimized algorithms for binning and feature selection

## License

This project is licensed under the MIT License - see the LICENSE file for details.
