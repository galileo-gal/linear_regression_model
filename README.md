It's impressive to see how much progress you've made on this project! Based on the code in `linear_regression.py`, the analysis in your Jupyter notebook, and the detailed notes in `PROJECT_LOG.md`, I've put together a comprehensive `README.md` for your portfolio.

I've made sure to highlight the "from scratch" nature of your work, as that's a key selling point for AI engineering roles. 🏠

---

# House Price Predictor - Linear Regression from Scratch

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Results](#2-key-results)
3. [Mathematical Implementation](#3-mathematical-implementation)
4. [Quick Start](#4-quick-start)
5. [Project Structure](#5-project-structure)
6. [Data Insights & Diagnostics](#6-data-insights--diagnostics)
7. [Model Performance](#7-model-performance)
8. [Lessons Learned](#8-lessons-learned)
9. [Future Improvements](#9-future-improvements)
10. [Author & Contact](#10-author--contact)

---

## 1. Project Overview

This project is a deep dive into the fundamentals of machine learning, focusing on predicting residential real estate prices using the **Kaggle House Prices - Advanced Regression Techniques** dataset. Rather than relying solely on high-level libraries, this project implements several regression algorithms from scratch to demonstrate a thorough understanding of optimization techniques like Gradient Descent and Coordinate Descent.

**Goal**: Build a robust predictive model while mastering the mathematical foundations of linear models and regularization.

## 2. Key Results

* **Best Model**: Linear Regression with Gradient Descent.
* **Performance**: Achieved an **R² score of 0.8449** on the validation set identical to Scikit-Learn's performance.
* **Data Processing**: Reduced target variable skewness from **1.88 to 0.12** using log transformation.
* **Convergence Mastery:** Successfully implemented a custom Gradient Descent that achieved a **99.98% cost reduction** within 1000 iterations.
* **Feature Engineering**: Selected 7 high-impact features, including `OverallQual` (0.79 correlation) and `GrLivArea` (0.71 correlation).

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

1. Download `train.csv` and `test.csv` from [Kaggle](https://www.google.com/search?q=https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place them in the `data/` folder.
2. Launch Jupyter Lab: `jupyter lab`
3. Open `notebooks/01_eda.ipynb` to view the full pipeline.

## 5. Project Structure

```text
house-price-predictor/
├── notebooks/          # Exploratory Data Analysis and experimentation
│   └── 01_eda.ipynb    # Main analysis and model training loop
├── src/                # Pure Python implementations
│   └── linear_regression.py  # Scratch-built ML classes
├── data/               # Dataset (not tracked in Git)
├── .gitignore          # Environment and data exclusions
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

```

## 6. Implementation Details




The core of this project is `linear_regression.py`, which contains custom implementations of:

* **LinearRegressionScratch**: Uses Gradient Descent with customizable learning rates and iterations. Includes gradient clipping for stability.
* **RidgeRegressionScratch**: Implements L2 regularization via the Normal Equation.
* **LassoRegressionScratch**: Implements L1 regularization using **Coordinate Descent** and a soft-thresholding operator.
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



## 8. Technologies Used

* **Core**: Python 3.12
* **Data Science**: NumPy, Pandas, Scikit-Learn (for preprocessing/validation)
* **Visualization**: Matplotlib, Seaborn
* **Tools**: Jupyter Lab, PyCharm, Git

## 9. Lessons Learned

* **Feature Scaling is Critical**: Gradient descent fails to converge effectively without standardization.
* **Log-Transformation Necessity:** House prices are naturally right-skewed. Applying a `log(Price)` transformation was critical to satisfying the linearity assumptions of the model.
* **Data Quality > Algorithm**: Exploratory data analysis accounted for 50% of the effort, and removing just two major outliers significantly improved model stability.
* **Target Transformation**: Log-transforming the skewed `SalePrice` was essential for meeting the linear assumption of the model.
* **The Multicollinearity Trap:** Discovered that highly correlated features (like `FullBath` and `GrLivArea`) can cause unstable, negative weights. This taught me the importance of feature selection over feature quantity.

## 10. Future Improvements

* **Advanced Engineering**: Combine bathroom features to reduce multicollinearity.
* **Interaction Terms**: Add features like `OverallQual × GrLivArea` to capture non-additive effects.
* **Segmented Models**: Develop separate models for luxury properties versus distressed properties to handle non-linear price breaks.

## 10. Author & Contact

**Abdullah Al Galib** I am a BSc. in Computer Science & Engineering student at **North South University (NSU)**, studying under the Department of Electrical and Computer Engineering (ECE).

* **Technical Foundation:** Strong background in Mathematics and Physics, having achieved **full UMS (300/300)** in AS Physics.
* **Interests:** Beyond AI, I am passionate about finance, budgeting, and cost optimization through analytical modeling.

* **GitHub**: [github.com/galileo-gal](https://www.google.com/search?q=https://github.com/galileo-gal)
* **LinkedIn**: [linkedin.com/in/galib3051](https://www.google.com/search?q=https://www.linkedin.com/in/galib3051/)

---

