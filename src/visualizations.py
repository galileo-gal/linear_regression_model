"""
Visualization utilities for house price prediction.

All plotting functions for model analysis and reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Set default style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_model_comparison(comparison_df: pd.DataFrame,
                          title: str = "Model Performance Comparison") -> None:
    """
    Bar chart comparing train/val/test R² across models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame from compare_models() with Train/Val/Test R² columns
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparison_df))
    width = 0.25

    ax.bar(x - width, comparison_df['Train R²'], width, label='Train R²', alpha=0.8, color='steelblue')
    ax.bar(x, comparison_df['Val R²'], width, label='Val R²', alpha=0.8, color='orange')
    ax.bar(x + width, comparison_df['Test R²'], width, label='Test R²', alpha=0.8, color='green')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0.80, 0.90])

    plt.tight_layout()
    plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     r2_score: float, title: str = "Actual vs Predicted") -> None:
    """
    Scatter plot of actual vs predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    r2_score : float
        R² score to display
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Log(Price)', fontsize=12)
    ax.set_ylabel('Predicted Log(Price)', fontsize=12)
    ax.set_title(f'{title} (R²={r2_score:.4f})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_residuals(y_pred: np.ndarray, residuals: np.ndarray,
                   title: str = "Residual Analysis") -> None:
    """
    4-panel residual analysis plot.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values
    residuals : np.ndarray
        Residual values (y_true - y_pred)
    title : str
        Main title
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residual plot
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[0, 0].axhline(0, color='red', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residual histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].grid(True, axis='y', alpha=0.3)

    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Absolute residuals
    axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.6, s=30, color='orange')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Absolute Residuals')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame,
                            title: str = "Feature Importance") -> None:
    """
    Horizontal bar chart of feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Abs_Weight' columns
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if w >= 0 else 'red' for w in importance_df['Weight']]

    ax.barh(importance_df['Feature'], importance_df['Abs_Weight'],
            color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Absolute Weight', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_cv_scores(cv_scores: np.ndarray, cv_type: str = "10-Fold CV") -> None:
    """
    Bar chart of cross-validation scores across folds.

    Parameters
    ----------
    cv_scores : np.ndarray
        Array of CV scores
    cv_type : str
        Type of CV (for title)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(1, len(cv_scores) + 1)
    ax.bar(x, cv_scores, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axhline(cv_scores.mean(), color='red', linestyle='--', lw=2,
               label=f'Mean: {cv_scores.mean():.4f}')

    ax.set_xlabel('Fold Number', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'{cv_type} Results', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([cv_scores.min() - 0.02, cv_scores.max() + 0.02])

    plt.tight_layout()
    plt.show()


def plot_learning_curve(cost_history: List[float],
                        title: str = "Learning Curve") -> None:
    """
    Plot training cost over iterations.

    Parameters
    ----------
    cost_history : list
        Cost values per iteration
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cost_history, linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add text annotations
    initial = cost_history[0]
    final = cost_history[-1]
    reduction = (1 - final / initial) * 100

    ax.text(0.05, 0.95, f'Initial: {initial:.4f}\nFinal: {final:.4f}\nReduction: {reduction:.2f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def create_summary_dashboard(model, data: Dict, evaluation: Dict) -> None:
    """
    Create comprehensive dashboard with all key plots.

    Parameters
    ----------
    model : trained model
        Final trained model
    data : dict
        Data dictionary from load_and_preprocess()
    evaluation : dict
        Evaluation dictionary from full_evaluation_report()
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Predictions
    ax1 = fig.add_subplot(gs[0, 0])
    y_pred = evaluation['test']['predictions']
    ax1.scatter(data['y_test'], y_pred, alpha=0.6, s=30)
    ax1.plot([data['y_test'].min(), data['y_test'].max()],
             [data['y_test'].min(), data['y_test'].max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'Predictions (R²={evaluation["test"]["r2"]:.3f})')
    ax1.grid(True, alpha=0.3)

    # 2. Residuals
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = evaluation['residuals']['residuals']
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Residual histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--')
    ax3.set_xlabel('Residuals')
    ax3.set_title('Residual Distribution')
    ax3.grid(True, axis='y', alpha=0.3)

    # 4. Feature importance
    ax4 = fig.add_subplot(gs[1, :])
    importance = evaluation['importance']
    ax4.barh(importance['Feature'], importance['Abs_Weight'], color='steelblue', alpha=0.8)
    ax4.set_xlabel('Absolute Weight')
    ax4.set_title('Feature Importance')
    ax4.invert_yaxis()
    ax4.grid(True, axis='x', alpha=0.3)

    # 5. CV scores
    ax5 = fig.add_subplot(gs[2, :2])
    cv_scores = evaluation['cv']['cv_scores']
    x = np.arange(1, len(cv_scores) + 1)
    ax5.bar(x, cv_scores, alpha=0.7, color='steelblue', edgecolor='black')
    ax5.axhline(cv_scores.mean(), color='red', linestyle='--', lw=2)
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('R² Score')
    ax5.set_title(f'10-Fold CV (Mean={cv_scores.mean():.3f})')
    ax5.grid(True, axis='y', alpha=0.3)

    # 6. Summary text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    summary_text = f"""
FINAL MODEL SUMMARY

Features: {len(data['feature_names'])}
{', '.join(data['feature_names'][:3])}...

Performance:
  CV R²:   {evaluation['cv']['mean']:.4f}
  Test R²: {evaluation['test']['r2']:.4f}
  RMSE:    {evaluation['test']['rmse']:.4f}
  MAE:     {evaluation['test']['mae']:.4f}

Status: ✓ Complete
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')

    fig.suptitle('House Price Prediction - Complete Analysis', fontsize=16, fontweight='bold')
    plt.show()


if __name__ == "__main__":
    print("Visualization module - import functions to use")
    print("Available functions:")
    print("  - plot_model_comparison()")
    print("  - plot_predictions()")
    print("  - plot_residuals()")
    print("  - plot_feature_importance()")
    print("  - plot_cv_scores()")
    print("  - plot_learning_curve()")
    print("  - create_summary_dashboard()")
