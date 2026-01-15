"""
Utility functions for house price prediction project.

Miscellaneous helper functions that don't fit in other modules.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Optional


def save_model(model, filepath: str) -> None:
    """
    Save trained model to disk using pickle.

    Parameters
    ----------
    model : object
        Trained model object
    filepath : str
        Path to save model (e.g., 'models/final_model.pkl')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to saved model

    Returns
    -------
    model : object
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    print(f"✓ Model loaded from {filepath}")
    return model


def save_results(results: Dict, filepath: str) -> None:
    """
    Save results dictionary to JSON.

    Parameters
    ----------
    results : dict
        Results to save
    filepath : str
        Path to save JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results_serializable = convert_numpy(results)

    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"✓ Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file.

    Parameters
    ----------
    filepath : str
        Path to JSON file

    Returns
    -------
    results : dict
        Loaded results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)

    print(f"✓ Results loaded from {filepath}")
    return results


def log_price_to_actual(log_price: np.ndarray) -> np.ndarray:
    """
    Convert log-transformed price back to actual price.

    Parameters
    ----------
    log_price : np.ndarray
        Log-transformed prices

    Returns
    -------
    actual_price : np.ndarray
        Actual prices in dollars
    """
    return np.exp(log_price)


def calculate_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate percentage error for each prediction.

    Parameters
    ----------
    y_true : np.ndarray
        True values (log scale)
    y_pred : np.ndarray
        Predicted values (log scale)

    Returns
    -------
    pct_error : np.ndarray
        Percentage errors
    """
    # Convert to actual prices
    actual_true = log_price_to_actual(y_true)
    actual_pred = log_price_to_actual(y_pred)

    # Calculate percentage error
    pct_error = ((actual_pred - actual_true) / actual_true) * 100

    return pct_error


def print_section_header(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.

    Parameters
    ----------
    title : str
        Section title
    width : int
        Total width of header
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def create_project_summary() -> Dict:
    """
    Create a summary dictionary of project specifications.

    Returns
    -------
    summary : dict
        Project summary
    """
    return {
        'project_name': 'House Price Predictor',
        'dataset': 'Kaggle House Prices',
        'total_samples': 1458,
        'features': {
            'original': 81,
            'selected': 7,
            'final': 6,
            'removed': ['FullBath (multicollinearity)']
        },
        'models_implemented': [
            'Linear Regression',
            'Polynomial Regression',
            'Ridge Regression',
            'Lasso Regression',
            'Elastic Net'
        ],
        'final_model': {
            'type': 'Linear Regression',
            'features': 6,
            'method': 'Gradient Descent',
            'test_r2': 0.8719
        },
        'key_techniques': [
            'Log transformation (target)',
            'Outlier removal',
            'Feature standardization',
            'Shuffled K-Fold CV',
            'Multicollinearity resolution'
        ]
    }


def format_price(price: float) -> str:
    """
    Format price as readable string.

    Parameters
    ----------
    price : float
        Price value

    Returns
    -------
    formatted : str
        Formatted price string (e.g., "$123,456")
    """
    return f"${price:,.0f}"


def get_model_equation(weights: np.ndarray, bias: float,
                       feature_names: list) -> str:
    """
    Generate string representation of model equation.

    Parameters
    ----------
    weights : np.ndarray
        Model weights
    bias : float
        Model bias
    feature_names : list
        Feature names

    Returns
    -------
    equation : str
        Model equation as string
    """
    equation = f"log(Price) = {bias:.4f}"

    for i, (feat, weight) in enumerate(zip(feature_names, weights)):
        sign = "+" if weight >= 0 else ""
        equation += f"\n           {sign} {weight:.4f} × {feat}"

    return equation


if __name__ == "__main__":
    print("Utility functions module")
    print("\nAvailable functions:")
    print("  - save_model() / load_model()")
    print("  - save_results() / load_results()")
    print("  - log_price_to_actual()")
    print("  - calculate_percentage_error()")
    print("  - print_section_header()")
    print("  - create_project_summary()")
    print("  - format_price()")
    print("  - get_model_equation()")

    # Global data cache
    _DATA_CACHE = None


def get_data(force_reload=False):
    """Get cached data or reload if needed"""
    global _DATA_CACHE

    if _DATA_CACHE is None or force_reload:
        from data_loader import load_and_preprocess
        _DATA_CACHE = load_and_preprocess()
        print("✓ Data loaded")
    else:
        print("✓ Using cached data")

    return _DATA_CACHE
