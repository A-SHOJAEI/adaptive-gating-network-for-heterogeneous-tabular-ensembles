"""Data preprocessing utilities."""

import logging
from typing import Optional, List, Union, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """Preprocessor for tabular data with support for numerical and categorical features.

    Handles missing values, categorical encoding, and feature scaling.

    Attributes:
        categorical_features: List of categorical feature names.
        numerical_features: List of numerical feature names.
        handle_missing: Strategy for handling missing values.
        scale_features: Whether to scale numerical features.
        scaler: Fitted StandardScaler instance.
        label_encoders: Dict of fitted LabelEncoder instances.
        feature_names: List of feature names from training data.
        is_fitted: Whether the preprocessor has been fitted.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        handle_missing: str = "mean",
        scale_features: bool = True,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            categorical_features: List of categorical feature names. Defaults to
                empty list if None.
            numerical_features: List of numerical feature names. Auto-detected
                if None.
            handle_missing: Strategy for handling missing values. Options are
                'mean', 'median', or 'drop'.
            scale_features: Whether to scale numerical features.

        Raises:
            ValueError: If handle_missing strategy is not supported.
        """
        if handle_missing not in ["mean", "median", "drop"]:
            raise ValueError(f"Invalid handle_missing: {handle_missing}")
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features
        self.handle_missing = handle_missing
        self.scale_features = scale_features

        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: dict = {}
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            X: Training features DataFrame.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X is empty or invalid.
        """
        if len(X) == 0:
            raise ValueError("X cannot be empty")

        logger.info("Fitting preprocessor on training data")

        # Auto-detect feature types if not specified
        if self.numerical_features is None:
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

        # Fit label encoders for categorical features
        for col in self.categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                # Handle missing values in categorical features
                valid_mask = X[col].notna()
                if valid_mask.any():
                    le.fit(X.loc[valid_mask, col])
                    self.label_encoders[col] = le

        # Fit scaler for numerical features
        if self.scale_features and self.numerical_features:
            self.scaler = StandardScaler()
            valid_numerical = [col for col in self.numerical_features if col in X.columns]
            if valid_numerical:
                self.scaler.fit(X[valid_numerical])

        self.feature_names = X.columns.tolist()
        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {len(self.feature_names)} features")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor.

        Args:
            X: Features DataFrame to transform.

        Returns:
            Transformed DataFrame.

        Raises:
            RuntimeError: If preprocessor has not been fitted.
            ValueError: If X is empty or has invalid structure.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        if len(X) == 0:
            raise ValueError("X cannot be empty")

        X_transformed = X.copy()

        # Handle missing values
        if self.handle_missing == "mean":
            for col in self.numerical_features:
                if col in X_transformed.columns and X_transformed[col].isna().any():
                    X_transformed[col].fillna(X_transformed[col].mean(), inplace=True)
        elif self.handle_missing == "median":
            for col in self.numerical_features:
                if col in X_transformed.columns and X_transformed[col].isna().any():
                    X_transformed[col].fillna(X_transformed[col].median(), inplace=True)
        elif self.handle_missing == "drop":
            X_transformed.dropna(inplace=True)

        # Encode categorical features
        for col, le in self.label_encoders.items():
            if col in X_transformed.columns:
                valid_mask = X_transformed[col].notna()
                if valid_mask.any():
                    # Handle unseen categories
                    known_classes = set(le.classes_)
                    X_transformed.loc[valid_mask, col] = X_transformed.loc[valid_mask, col].apply(
                        lambda x: x if x in known_classes else le.classes_[0]
                    )
                    X_transformed.loc[valid_mask, col] = le.transform(
                        X_transformed.loc[valid_mask, col]
                    )

        # Scale numerical features
        if self.scaler is not None:
            valid_numerical = [col for col in self.numerical_features if col in X_transformed.columns]
            if valid_numerical:
                X_transformed[valid_numerical] = self.scaler.transform(X_transformed[valid_numerical])

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data in one step.

        Args:
            X: Features DataFrame.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If X is empty or invalid.
        """
        return self.fit(X).transform(X)


def compute_feature_complexity(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Compute per-sample feature complexity indicators.

    This is a key innovation: we compute complexity metrics for each sample
    to guide the gating network's routing decisions.

    Args:
        X: Features DataFrame or numpy array of shape (n_samples, n_features).

    Returns:
        Array of complexity scores per sample of shape (n_samples,). Scores
        are normalized to have mean 0 and standard deviation 1.

    Raises:
        ValueError: If X is empty or has insufficient features.
    """
    if len(X) == 0:
        raise ValueError("X cannot be empty")
    # Convert to numpy array to ensure we work with arrays throughout
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    complexity_scores = np.zeros(len(X_array))

    # Feature interaction complexity: variance of pairwise products
    if X_array.shape[1] >= 2:
        for i in range(min(5, X_array.shape[1] - 1)):
            for j in range(i + 1, min(5, X_array.shape[1])):
                interaction = X_array[:, i] * X_array[:, j]
                complexity_scores += np.abs(interaction - interaction.mean())

    # Non-linearity indicator: deviation from linear projection
    if X_array.shape[1] >= 3:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(3, X_array.shape[1]))
        X_pca = pca.fit_transform(X_array)
        reconstruction = pca.inverse_transform(X_pca)
        reconstruction_error = np.mean((X_array - reconstruction) ** 2, axis=1)
        complexity_scores += reconstruction_error

    # Normalize scores to [0, 1] range
    if complexity_scores.std() > 0:
        # First standardize
        complexity_scores = (complexity_scores - complexity_scores.mean()) / complexity_scores.std()
        # Then map to [0, 1] using sigmoid-like transformation
        complexity_scores = 1 / (1 + np.exp(-complexity_scores))
    else:
        complexity_scores = np.full_like(complexity_scores, 0.5)

    return complexity_scores
