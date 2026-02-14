"""
Adaptive Gating Network for Heterogeneous Tabular Ensembles.

A meta-learning approach that trains a gating network to dynamically weight
heterogeneous base models on a per-sample basis using learned feature complexity indicators.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_gating_network_for_heterogeneous_tabular_ensembles.models.model import (
    AdaptiveGatingEnsemble,
)

__all__ = ["AdaptiveGatingEnsemble"]
