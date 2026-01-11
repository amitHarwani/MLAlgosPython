import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X): # X: (n_samples, n_features)
        # mean
        self.mean = np.mean(X, axis=0) # (n_features, )
        X = X - self.mean # (n_samples, n_features)
        # covariance matrix: transpose because cov function takes in an array where columns are samples.
        cov = np.cov(X.T) # (n_features, n_features)

        # eigenvectors and values: the function returns eigenvectors as column vectors in the matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T # each row is an eigenvector now

        # sort the eigenvalues and get the values in descending order
        idxs = np.argsort(eigenvalues)[::-1] 
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # keep first k eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # Projecting the data
        X = X - self.mean
        #X: (n_samples, n_features), components: 
        return np.dot(X, self.components.T)

