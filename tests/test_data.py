"""Tests for data loading and preprocessing."""

import pytest
import numpy as np
import pandas as pd

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.loader import (
    generate_synthetic_dataset,
    load_dataset,
    split_dataset,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
    TabularPreprocessor,
    compute_feature_complexity,
)


def test_generate_synthetic_dataset(random_seed):
    """Test synthetic dataset generation."""
    X, y = generate_synthetic_dataset(
        n_samples=1000, n_features=10, n_classes=2, random_state=random_seed
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 1000
    assert X.shape[1] == 10
    assert len(y) == 1000
    assert len(np.unique(y)) == 2


def test_load_synthetic_dataset(random_seed):
    """Test loading synthetic dataset."""
    X, y = load_dataset("synthetic_heterogeneous_benchmark", random_state=random_seed)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) > 0
    assert len(X) == len(y)


def test_split_dataset(sample_data, random_seed):
    """Test dataset splitting."""
    X, y = sample_data

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, test_size=0.2, val_size=0.1, random_state=random_seed
    )

    total_samples = len(X_train) + len(X_val) + len(X_test)
    assert total_samples == len(X)

    # Check no data leakage
    assert len(set(X_train.index) & set(X_test.index)) == 0
    assert len(set(X_train.index) & set(X_val.index)) == 0
    assert len(set(X_val.index) & set(X_test.index)) == 0


def test_tabular_preprocessor_fit_transform(sample_data):
    """Test preprocessor fit and transform."""
    X, y = sample_data

    preprocessor = TabularPreprocessor(scale_features=True)
    X_transformed = preprocessor.fit_transform(X)

    assert isinstance(X_transformed, pd.DataFrame)
    assert X_transformed.shape == X.shape
    assert preprocessor.is_fitted


def test_tabular_preprocessor_transform_without_fit(sample_data):
    """Test that transform fails without fit."""
    X, y = sample_data

    preprocessor = TabularPreprocessor(scale_features=True)

    with pytest.raises(RuntimeError):
        preprocessor.transform(X)


def test_compute_feature_complexity(sample_data):
    """Test feature complexity computation."""
    X, y = sample_data

    complexity = compute_feature_complexity(X)

    assert isinstance(complexity, np.ndarray)
    assert len(complexity) == len(X)
    assert not np.any(np.isnan(complexity))
