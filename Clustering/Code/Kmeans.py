import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv("./clustering/data/Mall_Customers.csv")
cols = dataset.iloc[:, [3, 4]].values

# find number of clusters in dataset using wcss matrix algorithm
wcss = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init="k-means++", random_state=42)
    k_means.fit(cols)
    wcss.append(k_means.inertia_)

# for wcss matrix

# plt.plot(range(1, 11), wcss)
# plt.show()


# train model
k_means = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pridct = k_means.fit_predict(cols)
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
plt.scatter(
    k_means.cluster_centers_[:, 0],
    k_means.cluster_centers_[:, 1],
    s=300,
    c="yellow",
    label="centroid",
)
print(k_means.cluster_centers_)
plt.title("clusters of customers")
plt.xlabel("annual income")
plt.ylabel("spending")
plt.show()