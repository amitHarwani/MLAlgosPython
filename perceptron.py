import numpy as np

class Perceptron:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
    
    def fit(self, X, y): # X: (num_samples, n_features)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) #(num_features,)
        self.bias = 0
        
        # Converting y to include only 1's and 0s'
        y_ = np.array([1 if i > 0 else 0 for i in y])
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias #(scalar,)
                y_predicted = self.activation_func(linear_output) # (scalar)

                update = self.lr * (y_[idx] - y_predicted) # (scalar)

                self.weights += update * x_i #(num_features,)
                self.bias += update
                

    def predict(self, X): # X: (num_test_samples, n_features)
        linear_output = X @ self.weights + self.bias # (num_test_samples,)
        y_predicted = self.activation_func(linear_output) # (num_test_samples)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)