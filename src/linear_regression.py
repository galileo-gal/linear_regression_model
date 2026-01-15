import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class LinearRegressionScratch:
    """
    Linear Regression with Gradient Descent

    Model: y = X @ w + b
    Cost: MSE = (1/m) * sum((y_pred - y)^2)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """Train the model using gradient descent"""
        m, n = X.shape

        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Compute cost
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            # Gradients
            dw = (1 / m) * (X.T @ (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # Clip gradients to prevent explosion



            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Cost: {cost:.4f}")

        return self

    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias

    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class PolynomialRegressionScratch:
    """
    Polynomial Regression using Normal Equation (closed-form solution)
    No gradient descent issues with high dimensions
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Transform to polynomial and solve with normal equation"""
        X_poly = self.poly_features.fit_transform(X)
        m, n = X_poly.shape

        # Add bias column
        X_b = np.c_[np.ones((m, 1)), X_poly]

        # Normal equation: θ = (X^T X)^-1 X^T y
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

        print(f"✓ Fitted with {n} polynomial features")
        return self

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return X_poly @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class PolynomialRegressionGradientDescent:
    """
    Polynomial Regression with Gradient Descent
    Uses optimized learning rate for high-dimensional features
    """

    def __init__(self, degree=2, learning_rate=0.06, n_iterations=2000):
        self.degree = degree
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """Transform to polynomial and train with gradient descent"""
        X_poly = self.poly_features.fit_transform(X)
        m, n = X_poly.shape

        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = X_poly @ self.weights + self.bias
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            dw = (1 / m) * (X_poly.T @ (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (i + 1) % 200 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Cost: {cost:.4f}")

        print(f"✓ Trained with {n} polynomial features")
        return self

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return X_poly @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class RidgeRegressionScratch:
    """Ridge Regression (L2 regularization) using Normal Equation"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]

        # Ridge: θ = (X^T X + αI)^-1 X^T y
        I = np.eye(n + 1)
        I[0, 0] = 0  # Don't regularize bias
        theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]
        print(f"✓ Ridge trained with alpha={self.alpha}")
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LassoRegressionScratch:
    """Lasso Regression (L1 regularization) using Coordinate Descent"""

    def __init__(self, alpha=1.0, n_iterations=1000, tol=1e-4):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None

    def _soft_threshold(self, rho, lambda_):
        """Soft thresholding operator for L1"""
        if rho < -lambda_:
            return rho + lambda_
        elif rho > lambda_:
            return rho - lambda_
        else:
            return 0

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = np.mean(y)

        # Coordinate descent
        for iteration in range(self.n_iterations):
            weights_old = self.weights.copy()

            for j in range(n):
                # Compute residual without feature j
                residual = y - (X @ self.weights + self.bias) + X[:, j] * self.weights[j]

                # Update weight j
                rho = np.dot(X[:, j], residual) / m
                self.weights[j] = self._soft_threshold(rho, self.alpha / m)

            # Update bias
            self.bias = np.mean(y - X @ self.weights)

            # Check convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                print(f"✓ Converged at iteration {iteration + 1}")
                break

        print(f"✓ Lasso trained with alpha={self.alpha}")
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class ElasticNetScratch:
    """
    Elastic Net Regression (L1 + L2 regularization) using Coordinate Descent.

    Combines Ridge (L2) and Lasso (L1) penalties:
    Cost = MSE + α * l1_ratio * |w| + α * (1-l1_ratio)/2 * w²

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Must be positive.

    l1_ratio : float, default=0.5
        Mix ratio between L1 and L2.
        - l1_ratio=0: Pure Ridge (L2)
        - l1_ratio=1: Pure Lasso (L1)
        - 0 < l1_ratio < 1: Elastic Net

    n_iterations : int, default=1000
        Maximum iterations for coordinate descent.

    tol : float, default=1e-4
        Convergence tolerance.
    """

    def __init__(self, alpha=1.0, l1_ratio=0.5, n_iterations=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None

    def _soft_threshold(self, rho, lambda_):
        """Soft thresholding for L1 penalty"""
        if rho < -lambda_:
            return rho + lambda_
        elif rho > lambda_:
            return rho - lambda_
        else:
            return 0

    def fit(self, X, y):
        """Fit Elastic Net using coordinate descent"""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = np.mean(y)

        # Precompute for efficiency
        l1_penalty = self.alpha * self.l1_ratio / m
        l2_penalty = self.alpha * (1 - self.l1_ratio) / m

        for iteration in range(self.n_iterations):
            weights_old = self.weights.copy()

            for j in range(n):
                # Compute residual without feature j
                residual = y - (X @ self.weights + self.bias) + X[:, j] * self.weights[j]

                # Coordinate update with L1 and L2
                rho = np.dot(X[:, j], residual) / m

                # Elastic Net update
                denominator = 1 + l2_penalty
                self.weights[j] = self._soft_threshold(rho, l1_penalty) / denominator

            # Update bias
            self.bias = np.mean(y - X @ self.weights)

            # Check convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                print(f"✓ Converged at iteration {iteration + 1}")
                break

        print(f"✓ Elastic Net trained (alpha={self.alpha}, l1_ratio={self.l1_ratio})")
        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def __repr__(self):
        return f"ElasticNetScratch(alpha={self.alpha}, l1_ratio={self.l1_ratio})"
