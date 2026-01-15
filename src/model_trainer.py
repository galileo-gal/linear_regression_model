"""
Model training utilities for house price prediction.

Functions for training all implemented models and returning results.
"""

import numpy as np
from typing import Dict, List
from linear_regression import (
    LinearRegressionScratch,
    PolynomialRegressionScratch,
    RidgeRegressionScratch,
    LassoRegressionScratch,
    ElasticNetScratch
)


def train_linear(X_train: np.ndarray, y_train: np.ndarray,
                 learning_rate: float = 0.01, n_iterations: int = 1000,
                 verbose: bool = False) -> LinearRegressionScratch:
    """Train linear regression model."""
    model = LinearRegressionScratch(learning_rate=learning_rate, n_iterations=n_iterations)

    if not verbose:
        # Suppress iteration prints
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        model.fit(X_train, y_train)
        sys.stdout = old_stdout
    else:
        model.fit(X_train, y_train)

    return model


def train_polynomial(X_train: np.ndarray, y_train: np.ndarray,
                     degree: int = 2) -> PolynomialRegressionScratch:
    """Train polynomial regression model (normal equation)."""
    model = PolynomialRegressionScratch(degree=degree)
    model.fit(X_train, y_train)
    return model


def train_ridge(X_train: np.ndarray, y_train: np.ndarray,
                alpha: float = 10.0) -> RidgeRegressionScratch:
    """Train ridge regression model."""
    model = RidgeRegressionScratch(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_lasso(X_train: np.ndarray, y_train: np.ndarray,
                alpha: float = 0.01, n_iterations: int = 1000) -> LassoRegressionScratch:
    """Train lasso regression model."""
    model = LassoRegressionScratch(alpha=alpha, n_iterations=n_iterations)
    model.fit(X_train, y_train)
    return model


def train_elastic_net(X_train: np.ndarray, y_train: np.ndarray,
                      alpha: float = 1.0, l1_ratio: float = 0.5,
                      n_iterations: int = 1000) -> ElasticNetScratch:
    """Train elastic net regression model."""
    model = ElasticNetScratch(alpha=alpha, l1_ratio=l1_ratio, n_iterations=n_iterations)
    model.fit(X_train, y_train)
    return model


def train_all_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              verbose: bool = False) -> Dict:
    """
    Train all baseline models (Days 3-5).

    Models:
    - Linear Regression
    - Polynomial Regression (degree=2)
    - Ridge Regression (alpha=10)
    - Lasso Regression (alpha=0.01)

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    verbose : bool
        Print training progress

    Returns
    -------
    results : dict
        Dictionary with model names as keys, containing:
        - model: trained model object
        - train_r2: training R²
        - val_r2: validation R²
    """
    print("=" * 60)
    print("TRAINING BASELINE MODELS")
    print("=" * 60)

    results = {}

    # 1. Linear Regression
    print("\n1. Linear Regression...")
    model_linear = train_linear(X_train, y_train, verbose=verbose)
    results['Linear'] = {
        'model': model_linear,
        'train_r2': model_linear.score(X_train, y_train),
        'val_r2': model_linear.score(X_val, y_val)
    }
    print(f"   Train R²: {results['Linear']['train_r2']:.4f}, Val R²: {results['Linear']['val_r2']:.4f}")

    # 2. Polynomial Regression
    print("\n2. Polynomial Regression (degree=2)...")
    model_poly = train_polynomial(X_train, y_train, degree=2)
    results['Polynomial'] = {
        'model': model_poly,
        'train_r2': model_poly.score(X_train, y_train),
        'val_r2': model_poly.score(X_val, y_val)
    }
    print(f"   Train R²: {results['Polynomial']['train_r2']:.4f}, Val R²: {results['Polynomial']['val_r2']:.4f}")

    # 3. Ridge Regression
    print("\n3. Ridge Regression (alpha=10)...")
    model_ridge = train_ridge(X_train, y_train, alpha=10.0)
    results['Ridge'] = {
        'model': model_ridge,
        'train_r2': model_ridge.score(X_train, y_train),
        'val_r2': model_ridge.score(X_val, y_val)
    }
    print(f"   Train R²: {results['Ridge']['train_r2']:.4f}, Val R²: {results['Ridge']['val_r2']:.4f}")

    # 4. Lasso Regression
    print("\n4. Lasso Regression (alpha=0.01)...")
    model_lasso = train_lasso(X_train, y_train, alpha=0.01)
    results['Lasso'] = {
        'model': model_lasso,
        'train_r2': model_lasso.score(X_train, y_train),
        'val_r2': model_lasso.score(X_val, y_val)
    }
    print(f"   Train R²: {results['Lasso']['train_r2']:.4f}, Val R²: {results['Lasso']['val_r2']:.4f}")

    print(f"\n{'=' * 60}")
    print("✓ All baseline models trained")

    return results


def train_final_model(X_train: np.ndarray, X_val: np.ndarray,
                      y_train: np.ndarray, y_val: np.ndarray) -> LinearRegressionScratch:
    """
    Train final optimized model on combined train+val data.

    Final model: Linear Regression (6 features, no FullBath)

    Returns
    -------
    model : LinearRegressionScratch
        Trained final model
    """
    print("=" * 60)
    print("TRAINING FINAL MODEL (Train+Val)")
    print("=" * 60)

    # Combine train and val
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])

    model = train_linear(X_combined, y_combined, verbose=True)

    train_r2 = model.score(X_combined, y_combined)
    print(f"\n✓ Final model trained")
    print(f"  Training R² (train+val): {train_r2:.4f}")

    return model


if __name__ == "__main__":
    # Test training functions
    from data_loader import load_and_preprocess

    data = load_and_preprocess()
    results = train_all_baseline_models(
        data['X_train_scaled'], data['y_train'],
        data['X_val_scaled'], data['y_val']
    )

    print("\nModel comparison:")
    for name, result in results.items():
        print(f"  {name:15s} Val R²: {result['val_r2']:.4f}")
