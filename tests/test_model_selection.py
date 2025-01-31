import pytest
from woe_scoring.core.model.selector import FeatureSelector

def test_feature_selector_initialization():
    selector = FeatureSelector(
        selection_type='rfe',
        random_state=42,
        class_weight='balanced',
        cv=3,
        n_jobs=1,
        max_vars=5,
        direction='forward',
        scoring='roc_auc',
        l1_exp_scale=4,
        l1_grid_size=20,
        iv_threshold=0.05
    )

    assert selector.selection_type == 'rfe'
    assert selector.max_vars == 5
    assert selector.iv_threshold == 0.05

def test_feature_selector_invalid_type():
    with pytest.raises(ValueError):
        FeatureSelector(
            selection_type='invalid',
            random_state=42,
            class_weight='balanced',
            cv=3,
            n_jobs=1,
            max_vars=5,
            direction='forward',
            scoring='roc_auc',
            l1_exp_scale=4,
            l1_grid_size=20,
            iv_threshold=0.05
        )

def test_feature_selection_methods(sample_data, woe_transformer):
    df, y = sample_data

    # Transform data
    woe_transformer.fit(df, y)
    woe_df = woe_transformer.transform(df)

    feature_names = list(woe_df.columns)

    # Test each selection method
    for selection_type in ['rfe', 'sfs', 'iv']:
        selector = FeatureSelector(
            selection_type=selection_type,
            random_state=42,
            class_weight='balanced',
            cv=3,
            n_jobs=1,
            max_vars=2,
            direction='forward',
            scoring='roc_auc',
            l1_exp_scale=4,
            l1_grid_size=20,
            iv_threshold=0.05
        )

        selected_features = selector.select(woe_df, y, feature_names)

        assert isinstance(selected_features, list)
        assert len(selected_features) <= 2  # max_vars=2
        assert all(f in feature_names for f in selected_features)
