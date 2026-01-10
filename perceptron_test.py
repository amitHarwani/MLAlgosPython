import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from perceptron import Perceptron

def accuracy(y_pred, y_true):
    return (y_pred == y_true).sum() / len(y_true)

X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = Perceptron(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy", accuracy(predictions, y_test))

# plt.figure() creates a new figure window (the canvas).
fig = plt.figure()
# fig.add_subplot(1, 1, 1) creates a grid of 1 row × 1 column and selects the first subplot
ax = fig.add_subplot(1, 1, 1)
""" Plots each training sample as a point in 2D feature space. 
X_train[:, 0] → feature 1 (x-axis)
X_train[:, 1] → feature 2 (y-axis)
c=y_train → colors each point by its class label
"""
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)

"""
Finds the minimum and maximum values of feature 0.
These will be the leftmost and rightmost x-values for drawing the decision boundary line.
"""
x0_1 = np.amin(X_train[:,0])
x0_2 = np.amax(X_train[:,0])

print("x0_1", x0_1, "x0_2", x0_2)
"""
For each x-value (x0_1, x0_2), it computes the corresponding y-value on the decision boundary.
Solving for x1 in : w0​x0​+w1​x1​+b=0
"""
x1_1 = (-model.weights[0] * x0_1 - model.bias) / model.weights[1]
x1_2 = (-model.weights[0] * x0_2 - model.bias) / model.weights[1]
print("x1_1", x1_1, "x1_2", x1_2)

"""
Draws a straight line between the two computed points, k is black line
"""
ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

"""
Finds min and max of feature 1 (y-values).
Expands the vertical range by 3 units above and below.
"""
ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin-3, ymax+3])
plt.show()
