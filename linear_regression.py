import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        y = y.reshape((n_samples, 1))

        for _ in range(self.n_iters):
            y_predicted = X @ self.weights + self.bias # (n_samples, 1)
            error = y_predicted - y # (n_samples, 1)

            dw = (2/n_samples)* (X.T@error) # (n_features, 1)
            db = (2/n_samples)* np.sum(error, 0) # (scalar)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return X @ self.weights + self.bias
