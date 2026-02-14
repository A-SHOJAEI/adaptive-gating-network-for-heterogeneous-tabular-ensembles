"""Data loading utilities for various datasets."""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, fetch_covtype
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    n_samples: int = 10000,
    n_features: int = 20,
    n_informative: int = 15,
    n_classes: int = 2,
    complexity_zones: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic heterogeneous benchmark dataset with controlled complexity zones.

    Creates a dataset where different regions have varying feature complexity,
    allowing some samples to be handled well by simple models and others requiring
    complex models.

    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        n_classes: Number of target classes
        complexity_zones: Number of distinct complexity regions
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    logger.info(f"Generating synthetic dataset with {n_samples} samples, {n_features} features")

    # Generate base dataset
    # Ensure n_redundant + n_informative + n_repeated <= n_features
    n_informative_adj = min(n_informative, n_features - 2)
    n_redundant = max(0, min(n_features - n_informative_adj - 2, n_features // 4))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative_adj,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=complexity_zones,
        flip_y=0.05,
        class_sep=0.8,
        random_state=random_state,
    )

    # Add complexity zones with different interaction patterns
    rng = np.random.RandomState(random_state)
    zone_assignments = rng.randint(0, complexity_zones, size=n_samples)

    for zone in range(complexity_zones):
        zone_mask = zone_assignments == zone
        zone_indices = np.where(zone_mask)[0]

        if len(zone_indices) == 0:
            continue

        if zone == 0:
            # Linear zone - simple linear relationships
            linear_feature = X[zone_indices, 0] * 0.5 + X[zone_indices, 1] * 0.3
            X[zone_indices, -1] = linear_feature

        elif zone == 1:
            # Non-linear zone - polynomial interactions
            X[zone_indices, -2] = (
                X[zone_indices, 0] ** 2 + X[zone_indices, 1] * X[zone_indices, 2]
            )

        else:
            # Complex zone - high-order interactions
            X[zone_indices, -3] = (
                X[zone_indices, 0] * X[zone_indices, 1] * X[zone_indices, 2]
            )

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    logger.info(
        f"Generated dataset shape: {X_df.shape}, "
        f"class distribution: {y_series.value_counts().to_dict()}"
    )

    return X_df, y_series


def load_dataset(
    dataset_name: str, data_dir: Optional[str] = None, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a dataset by name.

    Supported datasets:
    - synthetic_heterogeneous_benchmark: Generated synthetic dataset
    - covertype: Forest covertype classification
    - adult_income: UCI Adult Income dataset (requires manual download)
    - heloc: HELOC credit default (requires manual download)

    Args:
        dataset_name: Name of the dataset to load
        data_dir: Directory containing downloaded datasets (optional)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features DataFrame, target Series)

    Raises:
        ValueError: If dataset name is not recognized
        FileNotFoundError: If required data files are not found
    """
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "synthetic_heterogeneous_benchmark":
        return generate_synthetic_dataset(random_state=random_state)

    elif dataset_name == "covertype":
        # Load covertype from sklearn
        data = fetch_covtype(download_if_missing=True)
        X = pd.DataFrame(data.data, columns=[f"feature_{i}" for i in range(data.data.shape[1])])
        y = pd.Series(data.target, name="target")
        # Binary classification: class 1 vs others
        y = (y == 1).astype(int)
        logger.info(f"Loaded covertype dataset: {X.shape}")
        return X, y

    elif dataset_name == "adult_income":
        if data_dir is None:
            raise ValueError("data_dir must be provided for adult_income dataset")
        data_path = Path(data_dir) / "adult.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Adult income dataset not found at {data_path}. "
                "Download from UCI ML Repository."
            )
        df = pd.read_csv(data_path)
        X = df.drop(columns=["income"])
        y = pd.Series((df["income"] == ">50K").astype(int), name="target")
        logger.info(f"Loaded adult_income dataset: {X.shape}")
        return X, y

    elif dataset_name == "heloc":
        if data_dir is None:
            raise ValueError("data_dir must be provided for heloc dataset")
        data_path = Path(data_dir) / "heloc_dataset.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                f"HELOC dataset not found at {data_path}. " "Download from FICO website."
            )
        df = pd.read_csv(data_path)
        X = df.drop(columns=["RiskPerformance"])
        y = pd.Series((df["RiskPerformance"] == "Bad").astype(int), name="target")
        logger.info(f"Loaded HELOC dataset: {X.shape}")
        return X, y

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: synthetic_heterogeneous_benchmark, covertype, adult_income, heloc"
        )


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    logger.info(
        f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
