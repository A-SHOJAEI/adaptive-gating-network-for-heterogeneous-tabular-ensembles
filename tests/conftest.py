"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@pytest.fixture
def sample_data():
    """Generate sample dataset for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def train_test_split(sample_data):
    """Create train/test split for testing."""
    X, y = sample_data
    split_idx = int(0.7 * len(X))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42
