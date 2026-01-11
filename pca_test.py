import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from pca import PCA

data = datasets.load_iris()
X = data.data
y = data.target

model = PCA(2)
model.fit(X)
projected = model.transform(X)

print('Shape of X:', X.shape)
print('Shape of transformed X: ', projected.shape)

x1 = projected[:, 0]
x2 = projected[:, 1]

plt.scatter(x1, x2, c=y, edgecolors='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.colorbar()
plt.show()
