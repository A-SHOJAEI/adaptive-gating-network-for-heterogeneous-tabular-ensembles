#!/usr/bin/env python
"""Prediction script for adaptive gating ensemble."""

import sys
import argparse
import logging
import json
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import joblib


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_input_data(input_path: str) -> pd.DataFrame:
    """
    Load input data from file.

    Args:
        input_path: Path to input CSV file

    Returns:
        DataFrame with input features
    """
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".json"):
        return pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")


def main() -> None:
    """Main prediction function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Make predictions with adaptive gating ensemble")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pkl",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data (CSV or JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--show-routing",
        action="store_true",
        help="Show which model was selected for each sample",
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Show prediction confidence scores",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    logger.info("Starting prediction")

    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        checkpoint = joblib.load(checkpoint_path)
        model = checkpoint["model"]
        preprocessor = checkpoint.get("preprocessor")

        # Load input data
        logger.info(f"Loading input data from {args.input}")
        X = load_input_data(args.input)
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")

        # Preprocess data
        if preprocessor is not None:
            logger.info("Preprocessing input data")
            X_processed = preprocessor.transform(X)
        else:
            X_processed = X

        # Make predictions
        logger.info("Generating predictions")
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)

        # Get confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)

        # Create results DataFrame
        results = pd.DataFrame({"prediction": predictions})

        # Add confidence scores if requested
        if args.show_confidence:
            results["confidence"] = confidence_scores
            # Add probabilities for each class
            for i in range(probabilities.shape[1]):
                results[f"prob_class_{i}"] = probabilities[:, i]

        # Add routing information if requested
        if args.show_routing:
            X_array = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed
            routing = model.gating_network.predict(X_array)
            routing_weights = model.gating_network.predict_weights(X_array)
            results["assigned_model"] = routing

            # Map model indices to names
            model_names = [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM",
            ]
            results["assigned_model_name"] = results["assigned_model"].apply(
                lambda x: model_names[int(x)] if int(x) < len(model_names) else f"Model {int(x)}"
            )

            # Add routing weights for each model
            for i in range(routing_weights.shape[1]):
                model_name = model_names[i] if i < len(model_names) else f"Model {i}"
                results[f"weight_{model_name}"] = routing_weights[:, i]

        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(predictions)}")
        print(f"Unique predictions: {len(np.unique(predictions))}")
        print(f"Prediction distribution:")
        for label, count in zip(*np.unique(predictions, return_counts=True)):
            print(f"  Class {label}: {count} ({count / len(predictions) * 100:.1f}%)")

        if args.show_confidence:
            print(f"\nAverage confidence: {confidence_scores.mean():.4f}")
            print(f"Min confidence: {confidence_scores.min():.4f}")
            print(f"Max confidence: {confidence_scores.max():.4f}")

        if args.show_routing:
            print(f"\nRouting distribution:")
            routing_counts = results["assigned_model_name"].value_counts()
            for model_name, count in routing_counts.items():
                print(f"  {model_name}: {count} ({count / len(results) * 100:.1f}%)")

        print("=" * 60)
        print(f"\nPredictions saved to {output_path}")

        # Print first few predictions as preview
        print("\nFirst 5 predictions:")
        print(results.head())

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
