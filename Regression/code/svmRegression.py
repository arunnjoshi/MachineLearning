from matplotlib.pyplot import plot
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dataset = pd.read_csv("./Regression/DataSets/Position_Salaries.csv")
pos = dataset.iloc[:, 1].values
sal = dataset.iloc[:, -1].values

# changing dimension of numpy array ,because fit method need 2d input
pos = pos.reshape(len(pos), 1)
sal = sal.reshape(len(sal), 1)

# feature scaling
sc_pos = StandardScaler()
pos = sc_pos.fit_transform(pos)

sc_sal = StandardScaler()
sal = sc_sal.fit_transform(sal)

#  train model
regressor = SVR(kernel="rbf")
regressor.fit(
    pos,
    sal.reshape(
        len(sal),
    ),
)

# scale prediction
pr = sc_pos.transform([[6.5]])
pri_sal = regressor.predict(pr)
# inverse transform the pri_sal value
pri_sal = sc_sal.inverse_transform(pri_sal)
print(pri_sal)

# plot the model perdition
pos = sc_pos.inverse_transform(pos)

pos_grid = np.arange(min(pos), max(pos), 0.1)
pos_grid = pos_grid.reshape(len(pos_grid), 1)

sal = sc_sal.inverse_transform(sal)

pri_sal_grid = sc_sal.inverse_transform(
    regressor.predict(sc_pos.fit_transform(pos_grid))
)

plt.scatter(pos, sal, color="red")
plt.plot(pos_grid, pri_sal_grid, color="green")
plt.title("sal vs postion")
plt.xlabel("position")
plt.ylabel("salary")
plt.show()