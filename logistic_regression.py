import numpy as np

class LogisticRegression:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y): # X: (n_samples, n_features), y: (n_samples,)
        # initialize 
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # (n_features,)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = (X @ self.weights) + self.bias # (n_samples,)
            y_predicted = self._sigmoid(linear_model) # (n_samples,)
            
            error = (y_predicted - y) # (n_samples,)
            dw = (1/n_samples) * (X.T @ error) # (n_features,)
            db = (1/n_samples) * (np.sum(error, axis=0)) # scalar

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def predict(self, X):
        linear_model = (X @ self.weights) + self.bias # (n_samples,)
        y_predicted = self._sigmoid(linear_model) # (n_samples,)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted] # classes
        return np.array(y_predicted_cls)