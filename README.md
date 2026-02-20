# Adaptive Gating Network for Heterogeneous Tabular Ensembles

A PyTorch-based meta-learning system that trains a gating network to dynamically route samples to the most appropriate base model based on learned complexity indicators. The system uses a custom complexity-aware loss function to prioritize correct routing for samples where model selection matters most.

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

Train the model with default configuration:

```bash
python scripts/train.py
```

Train with custom configuration:

```bash
python scripts/train.py --config configs/ablation.yaml
```

Evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pkl
```

Make predictions:

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pkl --input data/samples.csv --output predictions.csv
```

## Architecture

The system consists of three components:

1. **Base Models**: Heterogeneous ensemble with different complexity handling capabilities
   - Logistic Regression (linear patterns)
   - Random Forest (moderate complexity)
   - XGBoost (complex interactions)
   - LightGBM (alternative boosting)

2. **Gating Network**: PyTorch MLP that predicts optimal model weights per sample
   - Learns from oracle assignments during training
   - Uses complexity-aware sample weighting
   - Supports custom loss functions with proper gradient flow

3. **Complexity Indicators**: Per-sample features capturing data complexity
   - Feature interaction variance
   - PCA reconstruction error (non-linearity measure)

## Key Innovation

**ComplexityAwareLoss**: A custom loss weighting scheme implemented in PyTorch that:
- Computes sample complexity indicators from feature interactions
- Weights routing errors based on complexity-model matching
- Prioritizes samples where base models disagree

Unlike traditional ensemble methods with fixed weights or simple voting, this approach adapts routing decisions per sample based on learned complexity characteristics. The gating network is trained with gradient-based optimization to predict oracle assignments, with sample weights that emphasize correct routing for complex samples where model selection matters most.

## Usage Example

```python
from adaptive_gating_network_for_heterogeneous_tabular_ensembles import AdaptiveGatingEnsemble
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.loader import load_dataset, split_dataset
from adaptive_gating_network_for_heterogeneous_tabular_ensembles.data.preprocessing import TabularPreprocessor

# Load and split data
X, y = load_dataset("synthetic_heterogeneous_benchmark")
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

# Preprocess
preprocessor = TabularPreprocessor(scale_features=True)
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# Train model
model = AdaptiveGatingEnsemble(
    gating_hidden_dims=[64, 32],
    gating_lr=0.001,
    use_complexity_loss=True,
    ensemble_method="weighted",
    random_state=42
)
model.fit(X_train, y_train, X_val, y_val)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Configuration

Hyperparameters are configurable via YAML files in `configs/`:

- `default.yaml`: Full model with complexity-aware loss
- `ablation.yaml`: Baseline without complexity-aware loss

Key parameters:
- `gating_hidden_dims`: Hidden layer sizes for gating network
- `gating_lr`: Learning rate for gating network training
- `use_complexity_loss`: Enable/disable complexity-aware sample weighting
- `ensemble_method`: Combination strategy (weighted, routing, stacking)

## Training Process

1. Train heterogeneous base models independently on training data
2. Determine optimal base model for each sample (oracle assignment)
3. Calculate per-sample complexity indicators
4. Train PyTorch gating network to predict oracle assignments with complexity-aware weights
5. At inference, combine base model predictions using learned gating weights

## Results

Performance on synthetic heterogeneous benchmark (10,000 samples, 20 features, binary classification):

| Metric | Value | Notes |
|--------|-------|-------|
| Test Accuracy | 0.8485 | Overall classification accuracy |
| Test F1 Score | 0.8485 | Weighted F1 across both classes |
| Macro F1 | 0.8485 | Unweighted average F1 |
| AUC | 0.9162 | Area under ROC curve |
| Gating Accuracy | 0.8445 | Routing correctness vs oracle |
| Routing Efficiency | 0.9145 | Fraction of optimal assignments |
| Complexity-Aware AUC | 0.9062 | AUC weighted by sample complexity |

Training metrics:
- Final training accuracy: 0.9574
- Final training F1: 0.9574
- Validation accuracy: 0.8500
- Validation F1: 0.8500

The gating network converged in 90 epochs with early stopping, achieving 88.71% validation accuracy in predicting oracle model assignments. The ensemble demonstrates strong performance with 84.85% test accuracy and high routing efficiency (91.45%).

*Results obtained on 2026-02-20 with random seed 42. Full metrics available in `results/results.json`.*

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Project Structure

```
adaptive-gating-network-for-heterogeneous-tabular-ensembles/
├── src/adaptive_gating_network_for_heterogeneous_tabular_ensembles/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model implementations (PyTorch gating network)
│   ├── training/       # Training loop with MLflow tracking
│   ├── evaluation/     # Metrics and analysis
│   └── utils/          # Configuration and helpers
├── tests/              # Unit tests
├── configs/            # YAML configuration files
├── scripts/            # Training, evaluation, and prediction scripts
├── checkpoints/        # Saved model checkpoints
└── results/            # Evaluation results and plots
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- mlflow >= 2.8.0

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See LICENSE file for details.
