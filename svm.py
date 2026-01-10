import numpy as np

class SVM:

    def __init__(self, lr = 0.001, lambda_param = 0.01, n_iters = 1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y): # y: (n_samples, ), X: (n_samples, n_features)
        # making sure y has +1 and -1 as classes
        y_ = np.where(y <= 0, -1, 1) #(n_samples,)

        n_samples, n_features = X.shape
        # Initialization
        self.weights = np.zeros(n_features) # (n_features,)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # SVM uses stochastic gradient descent
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (x_i @ self.weights + self.bias) >= 1 
                if condition:
                    # Update, 2*self.lambda_param*self.weights => derivative
                    self.weights -= self.lr * (2*self.lambda_param*self.weights)
                else:
                    self.weights -= self.lr * (2*self.lambda_param*self.weights - (y_[idx] * x_i))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = (X @ self.weights) + self.bias
        return np.sign(linear_output)