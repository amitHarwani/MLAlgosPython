import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from naive_bayes import NaiveBayes

def accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = NaiveBayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Classification Accuracy: ", accuracy(predictions, y_test))