import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Iris dataset
iris =  datasets.load_iris();

X, y = iris.data, iris.target

# Training and test labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

print("X_train shape", X_train.shape) # (120, 4)
print("y_train shape", y_train.shape) # (120,) 
print("unique classes", np.unique(y_train))

model = KNN(k=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predictions", predictions)
print("y_test     ", y_test)
acc = ((predictions == y_test).sum() / len(y_test)) * 100
print("Accuracy", acc)