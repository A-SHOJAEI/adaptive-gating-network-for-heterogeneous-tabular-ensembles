"""Custom model components including gating network and complexity-aware loss."""

import logging
from typing import Optional, List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GatingMLP(nn.Module):
    """PyTorch MLP for gating network.

    Attributes:
        layers: Sequential layers of the MLP.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], n_models: int) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            n_models: Number of output classes (models).
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_models))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output logits of shape (batch_size, n_models).
        """
        return self.layers(x)


class GatingNetwork(BaseEstimator, ClassifierMixin):
    """Neural network that learns to route samples to appropriate base models.

    This implements a PyTorch-based gating network that supports custom loss
    functions with sample weighting, enabling proper complexity-aware routing.

    Attributes:
        n_models: Number of base models to route between.
        hidden_dims: Hidden layer dimensions for the MLP.
        learning_rate: Learning rate for optimization.
        max_iter: Maximum training iterations.
        batch_size: Batch size for training.
        random_state: Random seed for reproducibility.
        device: Computation device (cuda/cpu).
        scaler: StandardScaler for input normalization.
        model: Underlying PyTorch MLP instance.
    """

    def __init__(
        self,
        n_models: int,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        max_iter: int = 200,
        batch_size: int = 64,
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the gating network.

        Args:
            n_models: Number of base models to route between.
            hidden_dims: Hidden layer dimensions. Defaults to [64, 32] if None.
            learning_rate: Learning rate for optimization.
            max_iter: Maximum training iterations.
            batch_size: Batch size for training.
            random_state: Random seed for reproducibility.
            device: Computation device. Defaults to auto-detect.

        Raises:
            ValueError: If n_models < 2.
        """
        if n_models < 2:
            raise ValueError(f"n_models must be >= 2, got {n_models}")

        self.n_models = n_models
        self.hidden_dims = hidden_dims or [64, 32]
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        np.random.seed(random_state)

        self.scaler = StandardScaler()
        self.model: Optional[GatingMLP] = None
        self.input_dim_: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        model_assignments: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "GatingNetwork":
        """Train the gating network to predict optimal model assignments.

        Args:
            X: Input features of shape (n_samples, n_features).
            model_assignments: Best model index for each sample (0 to n_models-1).
            sample_weight: Optional sample weights for complexity-aware loss.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If X or model_assignments are invalid.
            RuntimeError: If training fails.
        """
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        if len(X) != len(model_assignments):
            raise ValueError("X and model_assignments must have same length")
        if np.any((model_assignments < 0) | (model_assignments >= self.n_models)):
            raise ValueError(f"model_assignments must be in range [0, {self.n_models})")

        logger.info(f"Training PyTorch gating network on {len(X)} samples")

        try:
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            self.input_dim_ = X_scaled.shape[1]

            # Initialize model
            self.model = GatingMLP(
                input_dim=self.input_dim_,
                hidden_dims=self.hidden_dims,
                n_models=self.n_models
            ).to(self.device)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.LongTensor(model_assignments).to(self.device)

            if sample_weight is not None:
                weight_tensor = torch.FloatTensor(sample_weight).to(self.device)
            else:
                weight_tensor = torch.ones(len(X)).to(self.device)

            # Training setup
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.CrossEntropyLoss(reduction='none')

            # Split into train/val for early stopping
            n_train = int(0.9 * len(X))
            indices = np.random.permutation(len(X))
            train_idx, val_idx = indices[:n_train], indices[n_train:]

            X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
            y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
            w_train, w_val = weight_tensor[train_idx], weight_tensor[val_idx]

            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10

            # Training loop
            for epoch in range(self.max_iter):
                self.model.train()

                # Mini-batch training
                n_batches = (len(X_train) + self.batch_size - 1) // self.batch_size
                train_loss = 0.0

                for i in range(n_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, len(X_train))

                    batch_X = X_train[start_idx:end_idx]
                    batch_y = y_train[start_idx:end_idx]
                    batch_w = w_train[start_idx:end_idx]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss_per_sample = criterion(outputs, batch_y)
                    loss = (loss_per_sample * batch_w).mean()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= n_batches

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss_per_sample = criterion(val_outputs, y_val)
                    val_loss = (val_loss_per_sample * w_val).mean().item()
                    val_acc = (val_outputs.argmax(1) == y_val).float().mean().item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 20 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.max_iter}: "
                        f"train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, "
                        f"val_acc={val_acc:.4f}"
                    )

            logger.info(
                f"Gating network training complete. "
                f"Best validation loss: {best_val_loss:.4f}"
            )

        except Exception as e:
            logger.error(f"Gating network training failed: {e}")
            raise RuntimeError(f"Training failed: {e}") from e

        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        """Predict soft weights for each model.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_models) with weights summing to 1.

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict_weights")

        try:
            self.model.eval()
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            with torch.no_grad():
                logits = self.model(X_tensor)
                weights = torch.softmax(logits, dim=1)

            return weights.cpu().numpy()

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Failed to predict weights: {e}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict best model for each sample.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Array of model indices of shape (n_samples,).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        weights = self.predict_weights(X)
        return np.argmax(weights, axis=1)


class ComplexityAwareLoss:
    """Custom loss function that penalizes routing errors based on sample complexity.

    This weights the gating network's loss by how well the sample's complexity
    matches the model's strengths, prioritizing correct routing for complex samples.

    Attributes:
        base_models_complexity: Complexity level each base model handles best.
    """

    def __init__(self, base_models_complexity: Optional[List[float]] = None) -> None:
        """Initialize complexity-aware loss.

        Args:
            base_models_complexity: Complexity level each base model handles best,
                where 0 = simple/linear and 1 = complex/non-linear. Defaults to
                [0.0, 0.3, 0.7, 0.9] if None.
        """
        self.base_models_complexity = base_models_complexity or [0.0, 0.3, 0.7, 0.9]

    def compute_routing_penalty(
        self,
        sample_complexity: np.ndarray,
        model_assignments: np.ndarray,
        predictions_correct: np.ndarray,
    ) -> np.ndarray:
        """Compute penalty for routing samples to suboptimal models.

        Higher penalties are assigned when:
        1. Prediction is incorrect AND complexity mismatch is high
        2. Complex samples are routed to simple models or vice versa

        Args:
            sample_complexity: Complexity score for each sample (0 to 1) of shape
                (n_samples,).
            model_assignments: Assigned model index for each sample of shape
                (n_samples,).
            predictions_correct: Boolean array indicating whether each prediction
                was correct, of shape (n_samples,).

        Returns:
            Penalty weights for each sample of shape (n_samples,).

        Raises:
            ValueError: If input arrays have mismatched lengths.
        """
        if len(sample_complexity) != len(model_assignments):
            raise ValueError("sample_complexity and model_assignments must have same length")
        if len(sample_complexity) != len(predictions_correct):
            raise ValueError("sample_complexity and predictions_correct must have same length")

        penalties = np.ones(len(sample_complexity))

        for i, (complexity, model_idx, correct) in enumerate(
            zip(sample_complexity, model_assignments, predictions_correct)
        ):
            if model_idx >= len(self.base_models_complexity):
                continue

            # Compute mismatch between sample complexity and model capability
            model_complexity = self.base_models_complexity[model_idx]
            complexity_mismatch = abs(complexity - model_complexity)

            # Higher penalty if prediction was wrong AND complexity mismatch was high
            if not correct:
                penalties[i] = 1.0 + 2.0 * complexity_mismatch
            else:
                # Small penalty even if correct, to encourage optimal routing
                penalties[i] = 1.0 + 0.1 * complexity_mismatch

        return penalties

    def compute_sample_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_predictions: List[np.ndarray],
        sample_complexity: np.ndarray,
    ) -> np.ndarray:
        """Compute sample weights for gating network training.

        Assigns higher weight to samples where:
        1. Base models disagree (routing decision matters more)
        2. Sample complexity is near decision boundaries
        3. Complexity mismatch with oracle assignment is high

        Args:
            X: Input features of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).
            base_predictions: List of predictions from each base model, each of
                shape (n_samples,).
            sample_complexity: Complexity score for each sample of shape
                (n_samples,).

        Returns:
            Sample weights for training of shape (n_samples,).

        Raises:
            ValueError: If input dimensions are inconsistent.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        if len(X) != len(sample_complexity):
            raise ValueError("X and sample_complexity must have same length")
        if not base_predictions:
            raise ValueError("base_predictions cannot be empty")

        n_samples = len(X)
        weights = np.ones(n_samples)

        try:
            # Check which models got each sample correct
            model_correctness = np.array(
                [(pred == y).astype(float) for pred in base_predictions]
            ).T

            # Higher weight for samples where models disagree
            # (std=0 means all agree, std=0.5 means maximum disagreement)
            agreement = model_correctness.std(axis=1)
            weights += 2.0 * agreement

            # Higher weight for samples at complexity boundaries
            # (where routing decision matters most)
            # Samples near 0.5 complexity are harder to route
            complexity_boundary_distance = np.abs(sample_complexity - 0.5)
            weights += (1.0 - 2.0 * complexity_boundary_distance)

            # Normalize to mean=1 to maintain scale
            weights = weights / weights.mean()

        except Exception as e:
            logger.warning(f"Failed to compute sample weights: {e}. Using uniform weights.")
            weights = np.ones(n_samples)

        return weights


class EnsembleCalibrator:
    """Calibrates ensemble predictions using temperature scaling.

    Improves probability estimates without changing model rankings.

    Attributes:
        method: Calibration method to use.
        temperature: Fitted temperature scaling parameter.
    """

    def __init__(self, method: str = "temperature") -> None:
        """Initialize calibrator.

        Args:
            method: Calibration method ('temperature' or 'isotonic').

        Raises:
            ValueError: If method is not supported.
        """
        if method not in ["temperature", "isotonic"]:
            raise ValueError(f"Unsupported method: {method}. Use 'temperature' or 'isotonic'.")
        self.method = method
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "EnsembleCalibrator":
        """Fit calibration parameters.

        Args:
            logits: Raw prediction scores of shape (n_samples, n_classes).
            y_true: True labels of shape (n_samples,).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if len(logits) == 0:
            raise ValueError("logits cannot be empty")
        if len(logits) != len(y_true):
            raise ValueError("logits and y_true must have same length")

        if self.method == "temperature":
            try:
                from scipy.optimize import minimize

                def temperature_loss(temp: float) -> float:
                    """Compute cross-entropy loss with temperature scaling."""
                    temp = max(temp, 0.01)  # Avoid division by zero
                    calibrated = logits / temp
                    probs = np.exp(calibrated) / np.exp(calibrated).sum(axis=1, keepdims=True)
                    # Cross-entropy loss
                    return -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-10))

                result = minimize(temperature_loss, x0=1.0, bounds=[(0.01, 10.0)])
                self.temperature = float(result.x[0])
                logger.info(f"Fitted temperature scaling: T={self.temperature:.4f}")

            except Exception as e:
                logger.warning(f"Temperature scaling failed: {e}. Using default T=1.0")
                self.temperature = 1.0

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration.

        Args:
            logits: Raw prediction scores of shape (n_samples, n_classes).

        Returns:
            Calibrated probabilities of shape (n_samples, n_classes).

        Raises:
            ValueError: If logits shape is invalid.
        """
        if len(logits) == 0:
            raise ValueError("logits cannot be empty")

        if self.method == "temperature":
            try:
                calibrated = logits / self.temperature
                probs = np.exp(calibrated) / np.exp(calibrated).sum(axis=1, keepdims=True)
                return probs
            except Exception as e:
                logger.warning(f"Calibration transform failed: {e}. Returning original logits.")
                return logits

        return logits
