"""Results analysis and visualization utilities."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: Optional[list] = None, save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Names of classes
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(history: Dict[str, list], save_path: Optional[str] = None) -> None:
    """
    Plot training history.

    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    if "train_accuracy" in history and "val_accuracy" in history:
        axes[0].plot(history["train_accuracy"], label="Train", marker="o")
        axes[0].plot(history["val_accuracy"], label="Validation", marker="s")
        axes[0].set_title("Accuracy over Training")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Plot F1 score
    if "train_f1" in history and "val_f1" in history:
        axes[1].plot(history["train_f1"], label="Train", marker="o")
        axes[1].plot(history["val_f1"], label="Validation", marker="s")
        axes[1].set_title("F1 Score over Training")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_routing_distribution(
    routing_stats: Dict[str, Any], model_names: Optional[list] = None, save_path: Optional[str] = None
) -> None:
    """
    Plot routing distribution across models.

    Args:
        routing_stats: Dictionary with routing statistics
        model_names: Names of base models
        save_path: Path to save figure
    """
    if "routing_distribution" not in routing_stats:
        logger.warning("No routing distribution found in stats")
        return

    distribution = routing_stats["routing_distribution"]
    models = sorted(distribution.keys())
    counts = [distribution[m] for m in models]

    if model_names is None:
        model_names = [f"Model {i}" for i in models]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(models)), counts, tick_label=model_names)
    plt.title("Sample Routing Distribution")
    plt.xlabel("Base Model")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")

    if "routing_entropy" in routing_stats:
        plt.text(
            0.02,
            0.98,
            f"Routing Entropy: {routing_stats['routing_entropy']:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Routing distribution plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def save_results_table(results: Dict[str, Any], save_path: str) -> None:
    """
    Save results as formatted table.

    Args:
        results: Dictionary of results
        save_path: Path to save table
    """
    df = pd.DataFrame([results]).T
    df.columns = ["Value"]
    df.index.name = "Metric"

    # Format numbers
    df["Value"] = df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))

    # Save to CSV
    csv_path = Path(save_path).with_suffix(".csv")
    df.to_csv(csv_path)
    logger.info(f"Results table saved to {csv_path}")

    # Also save as formatted text
    txt_path = Path(save_path).with_suffix(".txt")
    with open(txt_path, "w") as f:
        f.write(df.to_string())
    logger.info(f"Results table saved to {txt_path}")
