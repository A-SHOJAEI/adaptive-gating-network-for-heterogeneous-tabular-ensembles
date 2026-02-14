"""Core adaptive gating ensemble model implementation."""

import logging
from typing import List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.components import (
    GatingNetwork,
    ComplexityAwareLoss,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
    compute_feature_complexity,
)

logger = logging.getLogger(__name__)


class AdaptiveGatingEnsemble(BaseEstimator, ClassifierMixin):
    """Adaptive gating network for heterogeneous tabular ensembles.

    Trains a meta-learner to dynamically route samples to the most appropriate
    base model based on learned feature complexity indicators.

    Attributes:
        base_models: List of base classifiers provided by user.
        gating_hidden_dims: Hidden layer dimensions for gating network.
        gating_lr: Learning rate for gating network.
        gating_iterations: Training iterations for gating network.
        use_complexity_loss: Whether to use complexity-aware loss.
        ensemble_method: How to combine predictions.
        random_state: Random seed for reproducibility.
        gating_network: Trained gating network instance.
        complexity_loss: Complexity-aware loss instance.
        base_models_: Fitted base model instances.
        classes_: Unique class labels.
        n_classes_: Number of classes.
    """

    def __init__(
        self,
        base_models: Optional[List[Any]] = None,
        gating_hidden_dims: Optional[List[int]] = None,
        gating_lr: float = 0.001,
        gating_iterations: int = 200,
        use_complexity_loss: bool = True,
        ensemble_method: str = "weighted",
        random_state: int = 42,
    ) -> None:
        """Initialize the adaptive gating ensemble.

        Args:
            base_models: List of base classifiers. If None, creates default
                heterogeneous set.
            gating_hidden_dims: Hidden layer dimensions for gating network.
                Defaults to [64, 32] if None.
            gating_lr: Learning rate for gating network.
            gating_iterations: Training iterations for gating network.
            use_complexity_loss: Whether to use complexity-aware loss.
            ensemble_method: How to combine predictions. Options are 'weighted',
                'routing', or 'stacking'.
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If ensemble_method is not supported.
        """
        if ensemble_method not in ["weighted", "routing", "stacking"]:
            raise ValueError(f"Invalid ensemble_method: {ensemble_method}")
        self.base_models = base_models
        self.gating_hidden_dims = gating_hidden_dims
        self.gating_lr = gating_lr
        self.gating_iterations = gating_iterations
        self.use_complexity_loss = use_complexity_loss
        self.ensemble_method = ensemble_method
        self.random_state = random_state

        self.gating_network: Optional[GatingNetwork] = None
        self.complexity_loss: Optional[ComplexityAwareLoss] = None
        self.base_models_: List[Any] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0

    def _create_default_base_models(self) -> List[Any]:
        """Create default heterogeneous base model ensemble.

        Returns:
            List of diverse base classifiers including Logistic Regression,
            Random Forest, XGBoost, and LightGBM.
        """
        logger.info("Creating default base models (linear, tree-based, gradient boosting)")

        return [
            # Simple linear model
            LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            ),
            # Random forest for moderate complexity
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            # XGBoost for complex interactions
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            # LightGBM for different boosting approach
            lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        ]

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "AdaptiveGatingEnsemble":
        """Fit the adaptive gating ensemble.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).
            X_val: Validation features (optional). Currently not used during
                base model training but reserved for future use.
            y_val: Validation labels (optional). Currently not used during
                base model training but reserved for future use.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X or y are empty or have mismatched lengths.
            RuntimeError: If model fitting fails.
        """
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have same length")

        logger.info(f"Fitting adaptive gating ensemble on {len(X)} samples")

        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Initialize base models if not provided
        if self.base_models is None:
            self.base_models_ = self._create_default_base_models()
        else:
            self.base_models_ = self.base_models

        # Convert to numpy arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Train base models
        logger.info(f"Training {len(self.base_models_)} base models")
        base_predictions = []

        for i, model in enumerate(self.base_models_):
            logger.info(f"Training base model {i}: {type(model).__name__}")
            try:
                model.fit(X_array, y_array)
                pred = model.predict(X_array)
                base_predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to train base model {i}: {e}")
                raise RuntimeError(f"Base model {i} training failed: {e}") from e

        # Compute feature complexity for each sample
        logger.info("Computing sample complexity indicators")
        sample_complexity = compute_feature_complexity(X)

        # Determine best model for each sample (oracle assignments)
        logger.info("Computing oracle model assignments")
        model_assignments = self._compute_oracle_assignments(
            y_array, base_predictions, sample_complexity
        )

        # Initialize and train gating network
        logger.info("Training gating network")
        self.gating_network = GatingNetwork(
            n_models=len(self.base_models_),
            hidden_dims=self.gating_hidden_dims,
            learning_rate=self.gating_lr,
            max_iter=self.gating_iterations,
            random_state=self.random_state,
        )

        # Compute sample weights using complexity-aware loss
        sample_weights = None
        if self.use_complexity_loss:
            self.complexity_loss = ComplexityAwareLoss(
                base_models_complexity=[0.0, 0.3, 0.7, 0.9]
            )
            sample_weights = self.complexity_loss.compute_sample_weights(
                X_array, y_array, base_predictions, sample_complexity
            )

        # Train gating network
        self.gating_network.fit(X_array, model_assignments, sample_weight=sample_weights)

        logger.info("Adaptive gating ensemble training complete")
        return self

    def _compute_oracle_assignments(
        self,
        y_true: np.ndarray,
        base_predictions: List[np.ndarray],
        sample_complexity: np.ndarray,
    ) -> np.ndarray:
        """
        Compute oracle model assignments based on correctness and complexity match.

        Args:
            y_true: True labels
            base_predictions: Predictions from each base model
            sample_complexity: Complexity score for each sample

        Returns:
            Array of best model indices for each sample
        """
        n_samples = len(y_true)
        n_models = len(base_predictions)
        assignments = np.zeros(n_samples, dtype=int)

        # Model complexity levels (manually assigned based on model type)
        model_complexities = np.linspace(0, 1, n_models)

        for i in range(n_samples):
            # Find which models got this sample correct
            correct_models = [j for j in range(n_models) if base_predictions[j][i] == y_true[i]]

            if correct_models:
                # Among correct models, choose the one with complexity closest to sample
                complexity_diffs = [
                    abs(sample_complexity[i] - model_complexities[j]) for j in correct_models
                ]
                best_idx = correct_models[np.argmin(complexity_diffs)]
            else:
                # If all wrong, assign to model with matching complexity
                complexity_diffs = [
                    abs(sample_complexity[i] - model_complexities[j]) for j in range(n_models)
                ]
                best_idx = np.argmin(complexity_diffs)

            assignments[i] = best_idx

        logger.info(
            f"Oracle assignments distribution: {dict(zip(*np.unique(assignments, return_counts=True)))}"
        )
        return assignments

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Array of class probabilities of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not hasattr(self, 'base_models_') or len(self.base_models_) == 0:
            raise RuntimeError("Model must be fitted before predict_proba")
        X_array = X.values if isinstance(X, pd.DataFrame) else X

        # Get base model predictions
        base_proba = []
        for model in self.base_models_:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_array)
            else:
                # For models without predict_proba, use one-hot encoding
                pred = model.predict(X_array)
                proba = np.eye(self.n_classes_)[pred]
            base_proba.append(proba)

        # Get gating weights
        gating_weights = self.gating_network.predict_weights(X_array)

        if self.ensemble_method == "weighted":
            # Weighted average of all models
            ensemble_proba = np.zeros((len(X_array), self.n_classes_))
            for i, proba in enumerate(base_proba):
                ensemble_proba += gating_weights[:, i:i+1] * proba

        elif self.ensemble_method == "routing":
            # Hard routing: use only the selected model
            selected_models = self.gating_network.predict(X_array)
            ensemble_proba = np.zeros((len(X_array), self.n_classes_))
            for i, model_idx in enumerate(selected_models):
                ensemble_proba[i] = base_proba[model_idx][i]

        else:  # stacking
            # Simple average (fallback)
            ensemble_proba = np.mean(base_proba, axis=0)

        return ensemble_proba

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Array of predicted labels of shape (n_samples,).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_routing_statistics(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Get statistics about routing decisions.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Dictionary with routing statistics including:
                - routing_distribution: Count of samples routed to each model.
                - routing_entropy: Entropy of routing distribution.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not hasattr(self, 'gating_network') or self.gating_network is None:
            raise RuntimeError("Model must be fitted before get_routing_statistics")
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        routing = self.gating_network.predict(X_array)

        stats = {
            "routing_distribution": dict(zip(*np.unique(routing, return_counts=True))),
            "routing_entropy": -np.sum(
                np.bincount(routing) / len(routing) * np.log(np.bincount(routing) / len(routing) + 1e-10)
            ),
        }

        return stats
