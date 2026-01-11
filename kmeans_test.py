import numpy as np
from sklearn import datasets
from kmeans import KMeans

X, y = datasets.make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

model = KMeans(K=clusters, max_iters=150)
predictions = model.predict(X)