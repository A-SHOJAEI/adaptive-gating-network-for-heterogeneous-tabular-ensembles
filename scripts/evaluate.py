#!/usr/bin/env python
"""Evaluation script for adaptive gating ensemble."""

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

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.utils.config import (
    load_config,
    set_random_seeds,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.loader import (
    load_dataset,
    split_dataset,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
    compute_feature_complexity,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.evaluation.metrics import (
    compute_all_metrics,
    compute_gating_accuracy,
    compute_routing_efficiency,
    compute_complexity_aware_auc,
    print_classification_report,
    compute_confusion_matrix,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.evaluation.analysis import (
    plot_confusion_matrix,
    plot_routing_distribution,
    save_results_table,
)


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


def main() -> None:
    """Main evaluation function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate adaptive gating ensemble")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pkl",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation")

    try:
        # Load configuration
        config = load_config(args.config)
        random_seed = config.get("random_seed", 42)
        set_random_seeds(random_seed)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = joblib.load(args.checkpoint)
        model = checkpoint["model"]
        preprocessor = checkpoint.get("preprocessor")

        # Load test data
        logger.info("Loading test dataset")
        data_config = config.get("data", {})
        X, y = load_dataset(
            dataset_name=data_config.get("dataset_name", "synthetic_heterogeneous_benchmark"),
            random_state=random_seed,
        )

        # Split to get test set
        _, _, X_test, _, _, y_test = split_dataset(
            X,
            y,
            test_size=data_config.get("test_size", 0.2),
            val_size=data_config.get("val_size", 0.1),
            random_state=random_seed,
        )

        # Preprocess test data
        if preprocessor is not None:
            logger.info("Preprocessing test data")
            X_test_processed = preprocessor.transform(X_test)
        else:
            X_test_processed = X_test

        # Generate predictions
        logger.info("Generating predictions")
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)

        # Compute standard metrics
        logger.info("Computing evaluation metrics")
        metrics = compute_all_metrics(y_test.values, y_pred, y_proba)

        # Compute gating-specific metrics
        logger.info("Computing gating-specific metrics")

        # Get base model predictions
        X_test_array = X_test_processed.values if isinstance(X_test_processed, pd.DataFrame) else X_test_processed
        base_predictions = []
        for base_model in model.base_models_:
            base_pred = base_model.predict(X_test_array)
            base_predictions.append(base_pred)

        # Get gating assignments
        gating_assignments = model.gating_network.predict(X_test_array)

        # Compute gating accuracy
        gating_acc = compute_gating_accuracy(y_test.values, base_predictions, gating_assignments)
        metrics["gating_accuracy"] = gating_acc

        # Compute routing efficiency
        routing_eff = compute_routing_efficiency(y_test.values, base_predictions, gating_assignments)
        metrics["routing_efficiency"] = routing_eff

        # Compute complexity-aware AUC
        sample_complexity = compute_feature_complexity(X_test)
        complexity_auc = compute_complexity_aware_auc(
            y_test.values, y_proba, sample_complexity, complexity_bins=3
        )
        metrics["complexity_aware_auc"] = complexity_auc

        # Get routing statistics
        routing_stats = model.get_routing_statistics(X_test_processed)

        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:30s}: {metric_value:.4f}")
        print("=" * 60)

        # Print classification report
        print("\nClassification Report:")
        print(print_classification_report(y_test.values, y_pred))

        # Save metrics to JSON
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # Save results table
        save_results_table(metrics, str(output_dir / "metrics_table"))

        # Plot confusion matrix
        if config.get("evaluation", {}).get("save_confusion_matrix", True):
            logger.info("Generating confusion matrix")
            cm = compute_confusion_matrix(y_test.values, y_pred)
            plot_confusion_matrix(
                cm,
                class_names=[f"Class {i}" for i in range(len(np.unique(y_test)))],
                save_path=str(output_dir / "confusion_matrix.png"),
            )

        # Plot routing distribution
        if config.get("evaluation", {}).get("compute_routing_stats", True):
            logger.info("Generating routing distribution plot")
            model_names = [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM",
            ]
            plot_routing_distribution(
                routing_stats,
                model_names=model_names[: len(model.base_models_)],
                save_path=str(output_dir / "routing_distribution.png"),
            )

        # Save per-sample results
        results_df = pd.DataFrame(
            {
                "true_label": y_test.values,
                "predicted_label": y_pred,
                "assigned_model": gating_assignments,
                "sample_complexity": sample_complexity,
            }
        )
        results_df.to_csv(output_dir / "per_sample_results.csv", index=False)
        logger.info(f"Per-sample results saved to {output_dir / 'per_sample_results.csv'}")

        print(f"\nEvaluation complete. Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
