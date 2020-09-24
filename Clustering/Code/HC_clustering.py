import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from scipy.cluster import hierarchy as hc
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv("./clustering/data/Mall_Customers.csv")
cols = dataset.iloc[:, [3, 4]].values


# dendrogram = hc.dendrogram(hc.linkage(cols, method="ward"))
# plot dendogram
# plt.show()


# train model
h_cluster = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
y_pridct = h_cluster.fit_predict(cols)
print(y_pridct)

plt.scatter(
    cols[y_pridct == 0, 0],
    cols[y_pridct == 0, 1],
    s=100,
    color="red",
    label="cluster 1",
)
plt.scatter(
    cols[y_pridct == 1, 0],
    cols[y_pridct == 1, 1],
    s=100,
    color="blue",
    label="cluster 2",
)
plt.scatter(
    cols[y_pridct == 2, 0],
    cols[y_pridct == 2, 1],
    s=100,
    color="green",
    label="cluster 3",
)
plt.scatter(
    cols[y_pridct == 3, 0],
    cols[y_pridct == 3, 1],
    s=100,
    color="pink",
    label="cluster 4",
)
plt.scatter(
    cols[y_pridct == 4, 0],
    cols[y_pridct == 4, 1],
    s=100,
    color="black",
    label="cluster 5",
)


plt.title("clusters of customers")
plt.xlabel("annual income")
plt.ylabel("spending")
plt.show()