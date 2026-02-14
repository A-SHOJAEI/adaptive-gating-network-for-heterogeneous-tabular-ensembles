"""Tests for model components."""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.model import (
    AdaptiveGatingEnsemble,
)
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.components import (
    GatingNetwork,
    ComplexityAwareLoss,
)


def test_gating_network_fit_predict(train_test_split, random_seed):
    """Test gating network training and prediction."""
    X_train, X_test, y_train, y_test = train_test_split

    # Convert to numpy for gating network
    X_train_np = X_train.values
    X_test_np = X_test.values

    # Create mock assignments (3 models)
    n_models = 3
    assignments = np.random.randint(0, n_models, size=len(X_train))

    gating_net = GatingNetwork(
        n_models=n_models, hidden_dims=[32, 16], learning_rate=0.01, max_iter=50, random_state=random_seed
    )

    gating_net.fit(X_train_np, assignments)
    weights = gating_net.predict_weights(X_test_np)

    assert weights.shape == (len(X_test), n_models)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)


def test_complexity_aware_loss():
    """Test complexity-aware loss computation."""
    loss = ComplexityAwareLoss(base_models_complexity=[0.0, 0.5, 1.0])

    sample_complexity = np.array([0.1, 0.5, 0.9])
    model_assignments = np.array([0, 1, 2])
    predictions_correct = np.array([True, False, True])

    penalties = loss.compute_routing_penalty(
        sample_complexity, model_assignments, predictions_correct
    )

    assert len(penalties) == len(sample_complexity)
    assert np.all(penalties >= 1.0)  # Penalties should be at least 1.0


def test_adaptive_gating_ensemble_fit(train_test_split, random_seed):
    """Test ensemble fitting."""
    X_train, X_test, y_train, y_test = train_test_split

    # Create small ensemble for testing
    base_models = [
        LogisticRegression(max_iter=100, random_state=random_seed),
        RandomForestClassifier(n_estimators=10, max_depth=3, random_state=random_seed),
    ]

    model = AdaptiveGatingEnsemble(
        base_models=base_models,
        gating_hidden_dims=[16],
        gating_iterations=50,
        random_state=random_seed,
    )

    model.fit(X_train, y_train)

    assert model.classes_ is not None
    assert len(model.base_models_) == 2
    assert model.gating_network is not None


def test_adaptive_gating_ensemble_predict(train_test_split, random_seed):
    """Test ensemble prediction."""
    X_train, X_test, y_train, y_test = train_test_split

    base_models = [
        LogisticRegression(max_iter=100, random_state=random_seed),
        RandomForestClassifier(n_estimators=10, max_depth=3, random_state=random_seed),
    ]

    model = AdaptiveGatingEnsemble(
        base_models=base_models,
        gating_hidden_dims=[16],
        gating_iterations=50,
        random_state=random_seed,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert all(pred in model.classes_ for pred in predictions)


def test_adaptive_gating_ensemble_predict_proba(train_test_split, random_seed):
    """Test ensemble probability prediction."""
    X_train, X_test, y_train, y_test = train_test_split

    base_models = [
        LogisticRegression(max_iter=100, random_state=random_seed),
        RandomForestClassifier(n_estimators=10, max_depth=3, random_state=random_seed),
    ]

    model = AdaptiveGatingEnsemble(
        base_models=base_models,
        gating_hidden_dims=[16],
        gating_iterations=50,
        random_state=random_seed,
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (len(X_test), model.n_classes_)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_adaptive_gating_ensemble_routing_stats(train_test_split, random_seed):
    """Test routing statistics extraction."""
    X_train, X_test, y_train, y_test = train_test_split

    model = AdaptiveGatingEnsemble(
        gating_hidden_dims=[16], gating_iterations=50, random_state=random_seed
    )

    model.fit(X_train, y_train)
    stats = model.get_routing_statistics(X_test)

    assert "routing_distribution" in stats
    assert "routing_entropy" in stats
    assert isinstance(stats["routing_distribution"], dict)
