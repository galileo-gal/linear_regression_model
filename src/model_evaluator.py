"""
Model evaluation utilities for house price prediction.

Functions for cross-validation, test evaluation, and model comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cross_validate_model(X_train: np.ndarray, X_val: np.ndarray,
                         y_train: np.ndarray, y_val: np.ndarray,
                         cv: int = 10, shuffle: bool = True,
                         random_state: int = 42) -> Dict:
    """
    Perform K-fold cross-validation.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Training and validation features
    y_train, y_val : np.ndarray
        Training and validation targets
    cv : int
        Number of folds
    shuffle : bool
        Shuffle data before splitting
    random_state : int
        Random seed

    Returns
    -------
    cv_results : dict
        Contains cv_scores, mean, std, min
    """
    # Combine train+val for CV
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])

    # Use sklearn for speed
    model = LinearRegression()
    kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state if shuffle else None)

    cv_scores = cross_val_score(model, X_combined, y_combined, cv=kf, scoring='r2')

    return {
        'cv_scores': cv_scores,
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'min': cv_scores.min(),
        'max': cv_scores.max()
    }


def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate model on test set.

    Parameters
    ----------
    model : regression model
        Trained model with predict() and score() methods
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets

    Returns
    -------
    test_results : dict
        Contains r2, rmse, mae, predictions
    """
    y_pred = model.predict(X_test)

    r2 = model.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred
    }


def compare_models(results: Dict, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare all trained models on test set.

    Parameters
    ----------
    results : dict
        Dictionary from train_all_baseline_models()
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test targets

    Returns
    -------
    comparison_df : pd.DataFrame
        Comparison table with all metrics
    """
    comparison = []

    for name, result in results.items():
        model = result['model']
        test_eval = evaluate_on_test(model, X_test, y_test)

        comparison.append({
            'Model': name,
            'Train R²': result['train_r2'],
            'Val R²': result['val_r2'],
            'Test R²': test_eval['r2'],
            'Test RMSE': test_eval['rmse'],
            'Test MAE': test_eval['mae']
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values('Test R²', ascending=False)

    return df


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model.

    Parameters
    ----------
    model : regression model
        Model with weights attribute
    feature_names : list
        Feature names

    Returns
    -------
    importance_df : pd.DataFrame
        Sorted by absolute weight
    """
    if not hasattr(model, 'weights'):
        raise ValueError("Model must have 'weights' attribute")

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': model.weights,
        'Abs_Weight': np.abs(model.weights)
    }).sort_values('Abs_Weight', ascending=False)

    return importance_df


def calculate_residuals(model, X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Calculate and analyze residuals.

    Parameters
    ----------
    model : regression model
    X : np.ndarray
        Features
    y : np.ndarray
        True targets

    Returns
    -------
    residuals_info : dict
        Statistics and residual array
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    return {
        'residuals': residuals,
        'mean': residuals.mean(),
        'std': residuals.std(),
        'min': residuals.min(),
        'max': residuals.max(),
        'median': np.median(residuals)
    }


def full_evaluation_report(model, data: Dict, cv_folds: int = 10) -> Dict:
    """
    Generate complete evaluation report for a model.

    Parameters
    ----------
    model : regression model
        Trained final model
    data : dict
        Data dictionary from load_and_preprocess()
    cv_folds : int
        Number of CV folds

    Returns
    -------
    report : dict
        Complete evaluation metrics
    """
    print("=" * 60)
    print("FULL EVALUATION REPORT")
    print("=" * 60)

    # Cross-validation
    print("\n1. Cross-Validation...")
    cv_results = cross_validate_model(
        data['X_train_scaled'], data['X_val_scaled'],
        data['y_train'], data['y_val'],
        cv=cv_folds, shuffle=True
    )
    print(f"   CV R²: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")

    # Test evaluation
    print("\n2. Test Set Evaluation...")
    test_results = evaluate_on_test(model, data['X_test_scaled'], data['y_test'])
    print(f"   Test R²: {test_results['r2']:.4f}")
    print(f"   Test RMSE: {test_results['rmse']:.4f}")
    print(f"   Test MAE: {test_results['mae']:.4f}")

    # Feature importance
    print("\n3. Feature Importance...")
    importance = get_feature_importance(model, data['feature_names'])
    print(importance.to_string(index=False))

    # Residuals
    print("\n4. Residual Analysis...")
    residuals = calculate_residuals(model, data['X_test_scaled'], data['y_test'])
    print(f"   Mean: {residuals['mean']:.4f}")
    print(f"   Std: {residuals['std']:.4f}")
    print(f"   Range: [{residuals['min']:.4f}, {residuals['max']:.4f}]")

    print(f"\n{'=' * 60}")
    print("✓ Evaluation complete")

    return {
        'cv': cv_results,
        'test': test_results,
        'importance': importance,
        'residuals': residuals
    }


if __name__ == "__main__":
    # Test evaluation functions
    from data_loader import load_and_preprocess
    from model_trainer import train_final_model

    data = load_and_preprocess()
    model = train_final_model(
        data['X_train_scaled'], data['X_val_scaled'],
        data['y_train'], data['y_val']
    )

    report = full_evaluation_report(model, data)
