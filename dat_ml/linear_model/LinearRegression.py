import numpy as np

# class LinearRegression:
#     def __init__(self, fit_intercept = True):
#         self.fit_intercept = fit_intercept
#         self.coef_ = None
#         self.intercept_ = None

#     def fit(self, X, y):
#         if self.fit_intercept:
#             X_b = np.c_[np.ones(len(X)), X]
#         else:
#             X_b = X
#         theta = np.linalg.pinv(X_b) @ y
#         if self.fit_intercept:
#             self.intercept_ = theta[0]
#             self.coef_ = theta[1:]
#         else:
#             self.intercept_ = 0.0
#             self.coef_ = theta
#         if self.coef_ is None:
#             raise RuntimeError("Model has not been fitted yet")
#         return self
    
#     def predict(self, X):
#         X = np.asarray(X, dtype=float)
#         return X @ self.coef_ + self.intercept_


class LinearRegression:
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y, learning_rate = 0.01, epochs = 10000):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")
        if X.shape[0] == 0 or X.shape[1] == 0:
                    raise ValueError("X must not be empty")
        if learning_rate <= 0:
             raise ValueError("Invalid learning_rate")
        if epochs <= 0:
                     raise ValueError("Invalid epochs")
        
        m, n = X.shape
        w = np.zeros(n)
        b = 0
        for epoch in range(epochs):
            y_pred = X @ w + b
            error = y_pred - y
            dw = (2/m) * (X.T @ error)
            db = (2/m) * np.sum(error)

            w -= learning_rate * dw
            if self.fit_intercept:
                b -= learning_rate * db

        self.coef_ = w
        self.intercept_ = b
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        return X @ self.coef_ + self.intercept_
