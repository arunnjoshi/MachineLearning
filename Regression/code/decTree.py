import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("./Regression/DataSets/Position_Salaries.csv")
pos = dataset.iloc[:, 1].values
sal = dataset.iloc[:, -1].values

# changing dimension of numpy array ,because fit method need 2d input
pos = pos.reshape(len(pos), 1)
sal = sal.reshape(len(sal), 1)

# train model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(pos, sal)

# predict  for input
sal_pri = regressor.predict([[6.5]])
print(sal_pri)

# plot the model
pos_grid = np.arange(min(pos), max(pos), 0.1)  # create list of pos from min to max
pos_grid = pos_grid.reshape(len(pos_grid), 1)  # change list to 2d array
sal_grid = regressor.predict(pos_grid)
plt.plot(pos_grid, sal_grid, color="green")
plt.scatter(pos, sal, color="red")
plt.title("sal vs postion")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()