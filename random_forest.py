import numpy as np
from decision_tree import DecisionTree
from collections import Counter

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    # get n_samples indices b/w 0 ... n_samples - 1, with replacement - can have duplicates and others get dropped
    idxc = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxc], y[idxc]

class RandomForest:

    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(self.min_samples_split,self.max_depth, self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # returns for each tree, prediction of each sample as an array
        # example: 3 trees 4 test samples [[1111] [0000] [1111]]
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # converting to [[101], [101], [101], [101]], each sample, predictions of each tree
        tree_preds = tree_preds.T
        return np.array([self._most_common_label(tree_pred) for tree_pred in tree_preds])
    
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]