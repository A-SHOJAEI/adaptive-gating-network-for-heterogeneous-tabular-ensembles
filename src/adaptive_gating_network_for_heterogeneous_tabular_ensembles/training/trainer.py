"""Training loop with learning rate scheduling and early stopping."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.model import (
    AdaptiveGatingEnsemble,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
    TabularPreprocessor,
)

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Trainer for adaptive gating ensemble with monitoring and checkpointing.
    """

    def __init__(
        self,
        model: AdaptiveGatingEnsemble,
        preprocessor: Optional[TabularPreprocessor] = None,
        checkpoint_dir: str = "checkpoints",
        patience: int = 10,
        min_delta: float = 0.001,
    ):
        """
        Initialize trainer.

        Args:
            model: AdaptiveGatingEnsemble instance
            preprocessor: Optional data preprocessor
            checkpoint_dir: Directory to save checkpoints
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        self.model = model
        self.preprocessor = preprocessor
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience
        self.min_delta = min_delta

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_accuracy": [],
            "train_f1": [],
            "val_accuracy": [],
            "val_f1": [],
        }
        self.best_val_score = -np.inf
        self.patience_counter = 0

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        track_mlflow: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the ensemble with monitoring.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            track_mlflow: Whether to track with MLflow

        Returns:
            Training history dictionary
        """
        logger.info("Starting ensemble training")

        # Initialize MLflow tracking if requested
        mlflow_client = None
        if track_mlflow:
            try:
                import mlflow

                mlflow.start_run()
                mlflow.log_params(
                    {
                        "gating_lr": self.model.gating_lr,
                        "gating_iterations": self.model.gating_iterations,
                        "ensemble_method": self.model.ensemble_method,
                        "use_complexity_loss": self.model.use_complexity_loss,
                        "n_train": len(X_train),
                        "n_val": len(X_val),
                    }
                )
                mlflow_client = mlflow
                logger.info("MLflow tracking enabled")
            except Exception as e:
                logger.warning(f"MLflow tracking failed: {e}. Continuing without MLflow.")
                mlflow_client = None

        # Preprocess data if preprocessor is provided
        if self.preprocessor is not None:
            logger.info("Preprocessing training data")
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_val_processed = self.preprocessor.transform(X_val)
        else:
            X_train_processed = X_train
            X_val_processed = X_val

        # Train the model
        try:
            self.model.fit(X_train_processed, y_train, X_val_processed, y_val)

            # Evaluate on training set
            train_pred = self.model.predict(X_train_processed)
            train_acc = accuracy_score(y_train, train_pred)
            train_f1 = f1_score(y_train, train_pred, average="weighted")

            # Evaluate on validation set
            val_pred = self.model.predict(X_val_processed)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average="weighted")

            # Update history
            self.history["train_accuracy"].append(train_acc)
            self.history["train_f1"].append(train_f1)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            logger.info(
                f"Training complete - Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Log to MLflow
            if mlflow_client is not None:
                try:
                    mlflow_client.log_metrics(
                        {
                            "train_accuracy": train_acc,
                            "train_f1": train_f1,
                            "val_accuracy": val_acc,
                            "val_f1": val_f1,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to log metrics to MLflow: {e}")

            # Save best model
            if val_f1 > self.best_val_score + self.min_delta:
                self.best_val_score = val_f1
                self.patience_counter = 0
                self._save_checkpoint("best_model.pkl")
                logger.info(f"New best model saved with val_f1: {val_f1:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {self.patience_counter}/{self.patience}"
                )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            if mlflow_client is not None:
                try:
                    mlflow.end_run()
                except Exception:
                    pass

        return self.history

    def _save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "history": self.history,
            "best_val_score": self.best_val_score,
        }
        joblib.dump(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = joblib.load(checkpoint_path)
        self.model = checkpoint["model"]
        self.preprocessor = checkpoint.get("preprocessor")
        self.history = checkpoint.get("history", {})
        self.best_val_score = checkpoint.get("best_val_score", -np.inf)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def save_history(self, filepath: str) -> None:
        """
        Save training history to JSON.

        Args:
            filepath: Path to save history
        """
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {filepath}")
