import numpy as np
from collections import Counter
class KNN:
    
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x): # x: (4,)
        # Distances
        dist = (((self.X_train - x)**2).sum(axis=1))**0.5 # (120,)

        # Sorted indices
        sorted_idx_upto_k = np.argpartition(dist, self.k)[:self.k]

        # Labels
        k_nearest_labels = [self.y_train[i] for i in sorted_idx_upto_k]
        
        # Majority count
        # It returns a list, which contains a tuple, where the first index is the element and the second index is the count
        return Counter(k_nearest_labels).most_common(1)[0][0] 
        
        

            