from numpy.lib.polynomial import poly
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

dataset = pd.read_csv("./Regression/DataSets/position_salaries.csv")
pos = dataset.iloc[:, 1].values
sal = dataset.iloc[:, -1].values

# changing dimension of numpy array ,because fit method need 2d input
pos = pos.reshape(len(pos), 1)
sal = sal.reshape(len(sal), 1)
# for linear req
lin_reg = LinearRegression()
lin_reg.fit(pos, sal)

# # for polynomial reg
poly_reg = PolynomialFeatures(degree=3)
pos_poly = poly_reg.fit_transform(pos)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(pos_poly, sal)

# plot
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
# sunplot one
ax1.scatter(pos, sal, color="red")
pos_grid = np.arange(min(pos), max(pos), 0.1)
pos_grid = pos_grid.reshape(len(pos_grid), 1)
sal_grid = lin_reg.predict(pos_grid)
ax1.plot(pos_grid, sal_grid, color="blue")
ax1.set_xlabel("postion")
ax1.set_ylabel("salary")
ax1.set_title("poly reg model")
# subplot two
pos_poly_grid = poly_reg.fit_transform(pos_grid)
ax2.scatter(pos, sal, color="yellow")
ax2.plot(pos_grid, lin_reg_2.predict(pos_poly_grid), color="green")
ax2.set_xlabel("postion")
ax2.set_ylabel("salary")

plt.show()