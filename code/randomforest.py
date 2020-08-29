from numpy.lib.polynomial import poly
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt

dataset = pd.read_csv("./datasets/position_salaries.csv")
pos = dataset.iloc[:, 1].values
sal = dataset.iloc[:, -1].values

# changing dimension of numpy array ,because fit method need 2d input
pos = pos.reshape(len(pos), 1)

# regressor for random forest model
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(pos, sal)
print(regressor.predict([[6.5]]))


# plot
pos_grid = np.arange(min(pos), max(pos), 0.1)  # create list of pos from min to max
pos_grid = pos_grid.reshape(len(pos_grid), 1)  # change list to 2d array
sal_grid = regressor.predict(pos_grid)
plt.plot(pos_grid, sal_grid, color="green")
plt.scatter(pos, sal, color="red")
plt.title("sal vs postion")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()