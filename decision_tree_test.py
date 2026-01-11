import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree

def accuracy(y_pred, y_true):
    return (y_pred == y_true).sum() / len(y_true)

data = datasets.load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = DecisionTree(max_depth=10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy: ", accuracy(predictions, y_test))