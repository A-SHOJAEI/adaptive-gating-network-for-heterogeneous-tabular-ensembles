#!/usr/bin/env python
"""Test script to verify installation and basic functionality."""

import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")

    try:
        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.model import (
            AdaptiveGatingEnsemble,
        )
        print("  ✓ AdaptiveGatingEnsemble")

        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.components import (
            GatingNetwork,
            ComplexityAwareLoss,
        )
        print("  ✓ GatingNetwork")
        print("  ✓ ComplexityAwareLoss")

        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.loader import (
            load_dataset,
            generate_synthetic_dataset,
        )
        print("  ✓ Data loaders")

        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
            TabularPreprocessor,
        )
        print("  ✓ TabularPreprocessor")

        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.training.trainer import (
            EnsembleTrainer,
        )
        print("  ✓ EnsembleTrainer")

        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.evaluation.metrics import (
            compute_all_metrics,
        )
        print("  ✓ Evaluation metrics")

        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.utils.config import (
            load_config,
        )
        print("  ✓ Config utilities")

        print("\nAll imports successful!")
        return True

    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return False


def test_config_loading():
    """Test that configuration files can be loaded."""
    print("\nTesting configuration loading...")

    try:
        import yaml

        config_dir = Path(__file__).parent.parent / "configs"

        # Test default config
        with open(config_dir / "default.yaml") as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Default config loaded")
        print(f"    - Use complexity loss: {config['model']['use_complexity_loss']}")

        # Test ablation config
        with open(config_dir / "ablation.yaml") as f:
            config = yaml.safe_load(f)
        print(f"  ✓ Ablation config loaded")
        print(f"    - Use complexity loss: {config['model']['use_complexity_loss']}")

        print("\nConfiguration files valid!")
        return True

    except Exception as e:
        print(f"\n✗ Config loading failed: {e}")
        return False


def test_quick_run():
    """Test a quick training run with minimal data."""
    print("\nTesting quick training run...")

    try:
        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.loader import (
            generate_synthetic_dataset,
            split_dataset,
        )
        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.model import (
            AdaptiveGatingEnsemble,
        )
        from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
            TabularPreprocessor,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        # Generate small dataset
        X, y = generate_synthetic_dataset(n_samples=200, n_features=10, random_state=42)
        print(f"  ✓ Generated dataset: {X.shape}")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            X, y, test_size=0.2, val_size=0.1, random_state=42
        )
        print(f"  ✓ Split dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # Preprocess
        preprocessor = TabularPreprocessor(scale_features=True)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test)
        print(f"  ✓ Preprocessed data")

        # Create small model for testing
        base_models = [
            LogisticRegression(max_iter=100, random_state=42),
            RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
        ]

        model = AdaptiveGatingEnsemble(
            base_models=base_models,
            gating_hidden_dims=[16],
            gating_iterations=20,
            random_state=42,
        )
        print(f"  ✓ Created model")

        # Train
        model.fit(X_train_proc, y_train, X_val_proc, y_val)
        print(f"  ✓ Model trained")

        # Predict
        predictions = model.predict(X_test_proc)
        probabilities = model.predict_proba(X_test_proc)
        print(f"  ✓ Generated predictions: {len(predictions)}")

        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")
        print(f"  ✓ Test accuracy: {acc:.4f}")
        print(f"  ✓ Test F1: {f1:.4f}")

        # Get routing stats
        routing_stats = model.get_routing_statistics(X_test_proc)
        print(f"  ✓ Routing distribution: {routing_stats['routing_distribution']}")

        print("\nQuick test run successful!")
        return True

    except Exception as e:
        print(f"\n✗ Quick run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Adaptive Gating Ensemble - Setup Test")
    print("=" * 60)

    success = True

    # Test imports
    if not test_imports():
        success = False
        print("\nSkipping remaining tests due to import failure.")
        print("Please install dependencies with: pip install -r requirements.txt")
        return

    # Test config loading
    if not test_config_loading():
        success = False

    # Test quick run
    if not test_quick_run():
        success = False

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("All tests passed! ✓")
        print("\nYou can now run:")
        print("  python scripts/train.py")
        print("  python scripts/evaluate.py")
        print("  python scripts/predict.py --input <file>")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
