import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    return ((y_true == y_pred).sum() / len(y_true)) * 100

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy", accuracy(y_test, predictions))