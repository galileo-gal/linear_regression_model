# House Price Predictor - Linear Regression from Scratch

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Results](#2-key-results)
3. [Mathematical Implementation](#3-mathematical-implementation)
4. [Quick Start](#4-quick-start)
5. [Project Structure](#5-project-structure)
6. [Data Insights & Diagnostics](#6-data-insights--diagnostics)
7. [Model Performance](#7-model-performance)
8. [Production Model & Confidence System](#8-production-model--confidence-system)
9. [Technologies & Tools](#9-technologies--tools)
10. [Lessons Learned](#10-lessons-learned)
11. [Completed Improvements](#11-completed-improvements)
12. [Author & Contact](#12-author--contact)

---

## 1. Project Overview

This project is a deep dive into the fundamentals of machine learning, focusing on predicting residential real estate prices using the **Kaggle House Prices - Advanced Regression Techniques** dataset. Rather than relying solely on high-level libraries, this project implements several regression algorithms from scratch to demonstrate a thorough understanding of optimization techniques like Gradient Descent and Coordinate Descent.

**Goal**: Build a robust predictive model while mastering the mathematical foundations of linear models and regularization.

## 2. Key Results

* **Champion Model**: Ensemble (Linear + XGBoost) with calibrated confidence intervals
* **Performance**: **R² = 0.8833** on test set (146 samples)
* **Coverage**: 88.4% of predictions fall within stated confidence intervals
* **Convergence**: Custom gradient descent achieved 99.98% cost reduction
* **Features**: 6 optimized features (removed multicollinear FullBath)
* **Validation**: All scratch implementations matched sklearn (<0.01 error)

### Model Evolution
- **Days 1-6**: Linear from scratch → 0.8718 R²
- **Day 9**: XGBoost hyperparameter tuning → 0.8740 R²
- **Day 10**: Ensemble stacking + calibration → 0.8833 R² (FINAL)

## 3. Mathematical Implementation
The core engine features two distinct optimization strategies built from the ground up:

### **Standard Linear Regression (Gradient Descent)**
Uses Batch Gradient Descent to minimize the Mean Squared Error (MSE).
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$



### **Lasso Regression (Coordinate Descent)**
To handle feature selection and L1 regularization, I implemented **Coordinate Descent** utilizing a **Soft-Thresholding Operator**. This allows the model to handle the non-differentiable nature of L1 penalties at zero.

$$S_{\lambda}(\rho) = \text{sign}(\rho) \cdot \max(0, |\rho| - \lambda)$$

## 4. Quick Start

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/galileo-gal/house-price-prediction.git
cd house-price-predictor

```


2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```


3. Install dependencies:
```bash
pip install -r requirements.txt

```



### Run the Analysis

1. Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place them in the `data/` folder.
2. Launch Jupyter Lab: `jupyter lab`
3. Open `notebooks/01_eda.ipynb` to view the full pipeline.

## 5. Project Structure

```text
house-price-predictor/
├── notebooks/               # Exploratory Data Analysis and experimentation
│   ├── 01_eda.ipynb        # Days 1-7: EDA, feature engineering, baseline models
│   └── 02_advanced.ipynb   # Days 8-10: XGBoost, ensemble, calibration
├── src/                     # Pure Python implementations
│   ├── linear_regression.py     # Scratch-built ML classes (Linear, Ridge, Lasso, Elastic Net)
│   ├── ensemble_model.py        # CalibratedPredictor class for production
│   ├── data_loader.py           # Data preprocessing pipeline
│   ├── model_trainer.py         # Training utilities
│   ├── model_evaluator.py       # Evaluation and cross-validation
│   ├── feature_engineering.py   # Feature engineering functions
│   ├── visualizations.py        # Plotting utilities
│   ├── utils.py                 # Helper functions
│   └── demo.py                  # Inference demo script
├── models/                  # Saved production models
│   ├── ensemble_production_v1.pkl  # Main ensemble model (Linear + XGBoost)
│   ├── linear_model.pkl            # Base linear regression model
│   └── xgboost_model.pkl           # Base XGBoost model
├── data/                    # Dataset (not tracked in Git)
│   ├── train.csv           # Kaggle training data
│   └── test.csv            # Kaggle test data
├── .gitignore              # Environment and data exclusions
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── PROJECT_LOG.md          # Detailed daily progress log

```

## 6. Implementation Details




The core of this project is `linear_regression.py`, which contains custom implementations of:

* **LinearRegressionScratch**: Uses Gradient Descent with customizable learning rates and iterations. Includes gradient clipping for stability.
* **RidgeRegressionScratch**: Implements L2 regularization via the Normal Equation.
* **LassoRegressionScratch**: Implements L1 regularization using **Coordinate Descent** and a soft-thresholding operator.
* **ElasticNetScratch**: Combines L1 and L2 regularization for flexible penalty mixing.
* **PolynomialRegressionGradientDescent**: Extends linear models to capture non-linear relationships.


The model utilizes Batch Gradient Descent to optimize weights. By building this from scratch, I ensured full control over the learning process.
The MathematicsThe loss is calculated using the Mean Squared Error (MSE) formula:$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

Why Scaling Mattered?\
Without Z-score Standardization, the features like YearBuilt (e.g., 1950) and FullBath (e.g., 1) would cause the gradient to "explode" due to the difference in scales. Normalizing everything to a mean of 0 and std of 1 ensured smooth convergence.

### **Real-World Price Impact**
A key value of this "from scratch" approach is interpretability. Based on the model's weights, we can quantify the percentage change in house price for every **1 Standard Deviation** increase in a feature:

| Feature      | Weight (Log) | Price Impact (%) |
|:-------------|:-------------|:-----------------|
| GrLivArea    | 0.1494       | +16.11%          |
| OverallQual  | 0.1288       | +13.75%          |
| YearBuilt    | 0.0864       | +9.03% (Appreciation) |
| FullBath     | -0.0207      | -2.05% (Multicollinearity) |


## 7. Model Performance

Through rigorous EDA and feature selection, the model effectively identifies the primary drivers of house value.

* **Validation Split**: 75% Train / 15% Val / 10% Test.
* **Strengths**: Strong performance on average-to-high value homes.
* **Limitations**:
* Overpredicts very cheap houses (<$50k) by roughly 50%.
* Identified multicollinearity between `FullBath` and `GrLivArea`.

### **Model Health Analysis**
By analyzing the residuals (the difference between actual and predicted prices), I identified that while the model is highly reliable for standard residential homes, it exhibits **Heteroscedasticity**. This "funnel shape" in the error distribution suggests that luxury properties ($400k+) carry unique non-linear variables that a simple linear model slightly underestimates.

## 8. Production Model & Confidence System

### Architecture
The final production model uses **ensemble stacking**:
1. **Base Model 1**: Linear Regression (from scratch, 6 features)
2. **Base Model 2**: XGBoost (300 trees, max_depth=4, learning_rate=0.05)
3. **Meta-Model**: Ridge blender (weights: 0.41×Linear + 0.66×XGBoost)

### Why Ensemble Won
- **Strategic advantage**: Beats both base models on 30.8% of houses
- **Handles outliers**: Averages out extreme errors from Linear and XGBoost
- **Low overfitting**: Only 0.63% gap between train and test
- **Captures non-linearity**: XGBoost fixes 7 houses where Linear failed by 20%+

### Confidence Scoring System
Predictions include **calibrated confidence intervals** based on model disagreement:

| Confidence | Criteria | Price Range | Coverage | Distribution |
|------------|----------|-------------|----------|--------------|
| **High** | Disagreement < 2.3% | ±14% | 80% | 25.3% of houses |
| **Medium** | Disagreement 2.3-6.7% | ±22% | 90% | 49.3% of houses |
| **Low** | Disagreement > 6.7% | ±31% | 95% | 25.3% of houses |

**Calibration Achievement**: 88.4% of actual prices fall within predicted ranges (industry standard: >85%)

### Example Output
```python
House Prediction:
  Price:       $245,000
  Confidence:  Medium
  Range:       $191,000 - $299,000
  Disagreement: 4.5% (models moderately uncertain)
```

### Production Deployment
- **Saved Models**: `models/ensemble_production_v1.pkl` (2.3 MB)
- **Load Time**: <0.5 seconds
- **Prediction Time**: Instant (<10ms)
- **Features Required**: 6 (OverallQual, GrLivArea, GarageCars, TotalBsmtSF, 1stFlrSF, YearBuilt)
- **Use Cases**: Portfolio valuation, automated appraisals, risk assessment

### System Design Philosophy
**Avoided model switching** based on uncertainty thresholds. Instead, the ensemble naturally handles uncertainty mathematically, while disagreement scores provide transparent confidence levels to users.

## 9. Technologies & Tools

* **Core**: Python 3.12
* **Data Science**: NumPy, Pandas, Scikit-Learn (for preprocessing/validation)
* **Advanced ML**: XGBoost, Ensemble Stacking
* **Visualization**: Matplotlib, Seaborn
* **Model Persistence**: Pickle (serialization)
* **Calibration**: Statistical threshold tuning (percentile-based)
* **Tools**: Jupyter Lab, PyCharm, Git

## 10. Lessons Learned

* **Feature Scaling is Critical**: Gradient descent fails to converge effectively without standardization.
* **Log-Transformation Necessity:** House prices are naturally right-skewed. Applying a `log(Price)` transformation was critical to satisfying the linearity assumptions of the model.
* **Data Quality > Algorithm**: Exploratory data analysis accounted for 50% of the effort, and removing just two major outliers significantly improved model stability.
* **Target Transformation**: Log-transforming the skewed `SalePrice` was essential for meeting the linear assumption of the model.
* **The Multicollinearity Trap:** Discovered that highly correlated features (like `FullBath` and `GrLivArea`) can cause unstable, negative weights. This taught me the importance of feature selection over feature quantity.
* **Ensemble strategic advantage**: Ensemble wins on outliers even if average error is slightly higher - R² measures variance explained, not just mean error.
* **Calibration essential**: Raw predictions need confidence intervals for production use - jumped from 52.7% to 88.4% coverage with data-driven thresholds.
* **System design matters**: Confidence scoring beats model switching - avoid arbitrary thresholds and discontinuities.
* **Tree models valuable**: XGBoost captured non-linearities (price discontinuities, quality premiums) that linear models missed.
* **Production readiness**: Serialization, validation, and UX (confidence intervals) are equally important as model accuracy.

## 11. Completed Improvements ✅

The following improvements were successfully implemented during the project:

* ✅ **Multicollinearity resolution**: Removed FullBath feature (negative weight artifact)
* ✅ **Cross-validation optimization**: Shuffled K-Fold reduced std by 55%
* ✅ **Advanced models tested**: Polynomial, Ridge, Lasso, Elastic Net, Random Forest, XGBoost
* ✅ **Ensemble stacking**: Combined Linear + XGBoost with meta-learner
* ✅ **Confidence calibration**: 88.4% coverage with 3-tier system
* ✅ **Production deployment**: Serialized models ready for API integration
* ✅ **Feature engineering**: Tested 15 features, validated that base 6 were optimal

### Potential Future Work
* **Temporal features**: Market trends, seasonality, economic indicators
* **Geographic expansion**: Train on multiple cities for generalization
* **Real-time integration**: Price tracking APIs, automated retraining pipeline
* **Web deployment**: Flask/FastAPI with interactive UI
* **Segmented models**: Separate predictors for luxury vs budget properties

## 12. Author & Contact

**Abdullah Al Galib** - BSc. in Computer Science & Engineering student at **North South University (NSU)**, studying under the Department of Electrical and Computer Engineering (ECE).

* **Technical Foundation:** Strong background in Mathematics and Physics, having achieved **full UMS (300/300)** in AS Physics.
* **Interests:** Beyond AI, I am passionate about finance, budgeting, and cost optimization through analytical modeling.

* **GitHub**: [github.com/galileo-gal](https://github.com/galileo-gal)
* **LinkedIn**: [linkedin.com/in/galib3051](https://linkedin.com/in/galib3051/)

---