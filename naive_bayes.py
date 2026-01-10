import numpy as np

class NaiveBayes:

    def fit(self, X, y): # X: (n_samples, n_features) | y: (n_samples,)
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # unique classes
        n_classes = len(self._classes)

        # initialization
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) # mean for each class and feature
        self._var = np.zeros((n_classes, n_features), dtype=np.float64) # variance for each class and feature
        self._priors = np.zeros(n_classes, dtype=np.float64) # prior probability of each class

        for c in self._classes:
            X_c = X[c==y] # the X's belonging to the class c
            # Computing the mean and variance for the class features
            self._mean[c,:] = np.mean(X_c, axis=0)
            self._var[c,:] = np.var(X_c, axis=0)

            # Prior class probability
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx]) # scalar
            class_conditional = np.sum(np.log(self._pdf(idx, x))) # scalar
            posterior = prior + class_conditional 
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)] # argmax returns the index of the max value/probability - idx of class

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx] #(n_features,)
        var = self._var[class_idx] # (n_features, )
        numerator = np.exp(- (x - mean)**2 / (2 * var)) # (n_features,)
        denominator = np.sqrt(2 * np.pi * var) # (n_features,)
        return numerator / denominator # (n_features,)


