"""
Data loading and preprocessing utilities for house price prediction.

Functions for loading raw data, cleaning, feature selection, splitting,
and standardization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def load_raw_data(filepath: str = '../data/train.csv') -> pd.DataFrame:
    """
    Load raw Kaggle house prices dataset.

    Parameters
    ----------
    filepath : str
        Path to CSV file

    Returns
    -------
    df : pd.DataFrame
        Raw dataset
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {df.shape[0]} samples, {df.shape[1]} features")
    return df


def select_features(df: pd.DataFrame, remove_fullbath: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select and extract features for modeling.

    Selected features based on correlation analysis:
    - OverallQual (0.79)
    - GrLivArea (0.71)
    - GarageCars (0.64)
    - TotalBsmtSF (0.61)
    - 1stFlrSF (0.61)
    - FullBath (0.56) - optional, has multicollinearity
    - YearBuilt (0.52)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    remove_fullbath : bool, default=True
        If True, exclude FullBath due to multicollinearity

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target variable (log-transformed SalePrice)
    """
    if remove_fullbath:
        feature_cols = ['OverallQual', 'GrLivArea', 'GarageCars',
                        'TotalBsmtSF', '1stFlrSF', 'YearBuilt']
    else:
        feature_cols = ['OverallQual', 'GrLivArea', 'GarageCars',
                        'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt']

    X = df[feature_cols].values
    y = np.log(df['SalePrice'].values)  # Log transformation

    print(f"✓ Selected {X.shape[1]} features")
    print(f"  Features: {feature_cols}")
    return X, y


def remove_outliers(X: np.ndarray, y: np.ndarray,
                    outlier_indices: list = [523, 1298]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove identified outlier samples.

    Outliers identified in EDA:
    - Index 523, 1298: Large houses with unusually low prices

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target variable
    outlier_indices : list
        Indices to remove

    Returns
    -------
    X_clean : np.ndarray
        Feature matrix without outliers
    y_clean : np.ndarray
        Target without outliers
    """
    mask = np.ones(len(X), dtype=bool)
    mask[outlier_indices] = False

    X_clean = X[mask]
    y_clean = y[mask]

    print(f"✓ Removed {len(outlier_indices)} outliers: {len(X)} → {len(X_clean)} samples")
    return X_clean, y_clean


def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.10,
               val_size: float = 0.15,
               random_state: int = 42) -> Tuple[np.ndarray, ...]:
    """
    Split data into train/validation/test sets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target variable
    test_size : float, default=0.10
        Proportion for test set
    val_size : float, default=0.15
        Proportion for validation set
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
        Split datasets
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation from train
    val_ratio = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    print(f"✓ Split complete:")
    print(f"  Train: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.1f}%)")
    print(f"  Val:   {X_val.shape[0]} ({X_val.shape[0] / len(X) * 100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_features(X_train: np.ndarray,
                         X_val: np.ndarray,
                         X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training set statistics.

    Formula: X_scaled = (X - mean) / std

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Feature matrices

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
        Standardized features
    X_mean, X_std : np.ndarray
        Training set statistics (for inverse transform)
    """
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)

    X_train_scaled = (X_train - X_mean) / X_std
    X_val_scaled = (X_val - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    print(f"✓ Features standardized (mean=0, std=1)")
    print(f"  Train mean: {X_train_scaled.mean(axis=0).round(3)}")
    print(f"  Train std:  {X_train_scaled.std(axis=0).round(3)}")

    return X_train_scaled, X_val_scaled, X_test_scaled, X_mean, X_std


def load_and_preprocess(filepath: str = '../data/train.csv',
                        remove_fullbath: bool = True,
                        remove_outliers_flag: bool = True) -> dict:
    """
    Complete preprocessing pipeline.

    Steps:
    1. Load raw data
    2. Select features
    3. Remove outliers
    4. Split into train/val/test
    5. Standardize features

    Parameters
    ----------
    filepath : str
        Path to CSV file
    remove_fullbath : bool, default=True
        Remove FullBath feature (multicollinearity)
    remove_outliers_flag : bool, default=True
        Remove identified outliers

    Returns
    -------
    data : dict
        Dictionary containing:
        - X_train_scaled, X_val_scaled, X_test_scaled
        - y_train, y_val, y_test
        - X_mean, X_std (for inverse transform)
        - feature_names
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load
    df = load_raw_data(filepath)

    # Select features
    X, y = select_features(df, remove_fullbath=remove_fullbath)

    # Remove outliers
    if remove_outliers_flag:
        X, y = remove_outliers(X, y)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Standardize
    X_train_scaled, X_val_scaled, X_test_scaled, X_mean, X_std = standardize_features(
        X_train, X_val, X_test
    )

    # Feature names
    if remove_fullbath:
        feature_names = ['OverallQual', 'GrLivArea', 'GarageCars',
                         'TotalBsmtSF', '1stFlrSF', 'YearBuilt']
    else:
        feature_names = ['OverallQual', 'GrLivArea', 'GarageCars',
                         'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt']

    print(f"\n✓ Preprocessing complete!")

    return {
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'X_mean': X_mean,
        'X_std': X_std,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    # Test the pipeline
    data = load_and_preprocess()
    print(f"\nData shapes:")
    print(f"  X_train: {data['X_train_scaled'].shape}")
    print(f"  Features: {data['feature_names']}")