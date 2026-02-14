#!/usr/bin/env python
"""Training script for adaptive gating ensemble."""

import sys
import argparse
import logging
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.utils.config import (
    load_config,
    set_random_seeds,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.loader import (
    load_dataset,
    split_dataset,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
    TabularPreprocessor,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.model import (
    AdaptiveGatingEnsemble,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.training.trainer import (
    EnsembleTrainer,
)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Raises:
        ValueError: If log_level is invalid.
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level_upper = log_level.upper()
    if log_level_upper not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Must be one of {valid_levels}")

    try:
        logging.basicConfig(
            level=getattr(logging, log_level_upper),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
        )
    except Exception as e:
        print(f"Warning: Failed to configure logging: {e}")
        logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Main training function.

    Trains an adaptive gating ensemble model with the specified configuration.

    Raises:
        FileNotFoundError: If config file or dataset not found.
        ValueError: If configuration is invalid.
        RuntimeError: If training fails.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train adaptive gating ensemble")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        raise
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}")
        raise

    # Setup logging
    log_config = config.get("logging", {})
    setup_logging(log_config.get("level", "INFO"))
    logger = logging.getLogger(__name__)
    logger.info("Starting adaptive gating ensemble training")

    # Set random seeds
    random_seed = config.get("random_seed", 42)
    set_random_seeds(random_seed)

    try:
        # Load data
        logger.info("Loading dataset")
        data_config = config.get("data", {})
        try:
            X, y = load_dataset(
                dataset_name=data_config.get("dataset_name", "synthetic_heterogeneous_benchmark"),
                random_state=random_seed,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}") from e

        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        # Split dataset
        logger.info("Splitting dataset")
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
                X,
                y,
                test_size=data_config.get("test_size", 0.2),
                val_size=data_config.get("val_size", 0.1),
                random_state=random_seed,
            )
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            raise RuntimeError(f"Dataset splitting failed: {e}") from e

        # Initialize preprocessor
        logger.info("Initializing preprocessor")
        preprocessing_config = config.get("preprocessing", {})
        try:
            preprocessor = TabularPreprocessor(
                handle_missing=preprocessing_config.get("handle_missing", "mean"),
                scale_features=preprocessing_config.get("scale_features", True),
            )
        except Exception as e:
            logger.error(f"Failed to initialize preprocessor: {e}")
            raise RuntimeError(f"Preprocessor initialization failed: {e}") from e

        # Initialize model
        logger.info("Initializing adaptive gating ensemble")
        model_config = config.get("model", {})
        try:
            model = AdaptiveGatingEnsemble(
                gating_hidden_dims=model_config.get("gating_hidden_dims", [64, 32]),
                gating_lr=model_config.get("gating_lr", 0.001),
                gating_iterations=model_config.get("gating_iterations", 200),
                use_complexity_loss=model_config.get("use_complexity_loss", True),
                ensemble_method=model_config.get("ensemble_method", "weighted"),
                random_state=random_seed,
            )
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

        # Initialize trainer
        logger.info("Initializing trainer")
        training_config = config.get("training", {})
        try:
            trainer = EnsembleTrainer(
                model=model,
                preprocessor=preprocessor,
                checkpoint_dir=training_config.get("checkpoint_dir", "checkpoints"),
                patience=training_config.get("patience", 10),
                min_delta=training_config.get("min_delta", 0.001),
            )
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise RuntimeError(f"Trainer initialization failed: {e}") from e

        # Train model
        logger.info("Starting training")
        try:
            history = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                track_mlflow=training_config.get("track_mlflow", True),
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Model training failed: {e}") from e

        # Save training history
        results_dir = Path(config.get("evaluation", {}).get("results_dir", "results"))
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_history(str(results_dir / "training_history.json"))
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")

        # Evaluate on test set
        logger.info("Evaluating on test set")
        try:
            X_test_processed = preprocessor.transform(X_test)
            test_pred = model.predict(X_test_processed)
            test_proba = model.predict_proba(X_test_processed)

            from sklearn.metrics import accuracy_score, f1_score, classification_report

            test_acc = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average="weighted")

            logger.info(f"Test Accuracy: {test_acc:.4f}")
            logger.info(f"Test F1 Score: {test_f1:.4f}")
            logger.info(f"\nClassification Report:\n{classification_report(y_test, test_pred)}")

            # Save test results
            import json

            test_results = {
                "test_accuracy": float(test_acc),
                "test_f1": float(test_f1),
                "config": args.config,
            }
            try:
                with open(results_dir / "test_results.json", "w") as f:
                    json.dump(test_results, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save test results: {e}")

            logger.info(f"Training complete. Results saved to {results_dir}")
            print(f"\nTraining complete!")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
            print(f"Best model saved to: checkpoints/best_model.pkl")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise RuntimeError(f"Test evaluation failed: {e}") from e

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
