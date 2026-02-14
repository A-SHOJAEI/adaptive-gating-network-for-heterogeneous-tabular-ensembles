"""Tests for training utilities."""

import pytest
import tempfile
from pathlib import Path

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.model import (
    AdaptiveGatingEnsemble,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.training.trainer import (
    EnsembleTrainer,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import (
    TabularPreprocessor,
)


def test_ensemble_trainer_init(random_seed):
    """Test trainer initialization."""
    model = AdaptiveGatingEnsemble(random_state=random_seed)
    preprocessor = TabularPreprocessor()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnsembleTrainer(
            model=model, preprocessor=preprocessor, checkpoint_dir=tmpdir, patience=5
        )

        assert trainer.model is model
        assert trainer.preprocessor is preprocessor
        assert trainer.patience == 5
        assert Path(tmpdir).exists()


def test_ensemble_trainer_train(train_test_split, random_seed):
    """Test trainer training."""
    X_train, X_test, y_train, y_test = train_test_split

    # Split X_test into val and test
    split_idx = len(X_test) // 2
    X_val = X_test.iloc[:split_idx]
    y_val = y_test.iloc[:split_idx]

    model = AdaptiveGatingEnsemble(gating_iterations=50, random_state=random_seed)
    preprocessor = TabularPreprocessor()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnsembleTrainer(
            model=model, preprocessor=preprocessor, checkpoint_dir=tmpdir, patience=5
        )

        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            track_mlflow=False,  # Disable MLflow for testing
        )

        assert "train_accuracy" in history
        assert "val_accuracy" in history
        assert len(history["train_accuracy"]) > 0


def test_ensemble_trainer_checkpoint_save_load(train_test_split, random_seed):
    """Test checkpoint saving and loading."""
    X_train, X_test, y_train, y_test = train_test_split

    split_idx = len(X_test) // 2
    X_val = X_test.iloc[:split_idx]
    y_val = y_test.iloc[:split_idx]

    model = AdaptiveGatingEnsemble(gating_iterations=50, random_state=random_seed)
    preprocessor = TabularPreprocessor()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = EnsembleTrainer(
            model=model, preprocessor=preprocessor, checkpoint_dir=tmpdir
        )

        # Train
        trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            track_mlflow=False,
        )

        # Check checkpoint exists
        checkpoint_path = Path(tmpdir) / "best_model.pkl"
        assert checkpoint_path.exists()

        # Load checkpoint
        new_trainer = EnsembleTrainer(
            model=AdaptiveGatingEnsemble(random_state=random_seed),
            preprocessor=TabularPreprocessor(),
            checkpoint_dir=tmpdir,
        )
        new_trainer.load_checkpoint("best_model.pkl")

        assert new_trainer.model is not None
        assert new_trainer.preprocessor is not None
