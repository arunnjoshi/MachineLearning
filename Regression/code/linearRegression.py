import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("./Regression/DataSets/Salary_Data.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pridct = regressor.predict(x_test)
x_pridct = regressor.predict(x_train)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

ax1.set_title("sal vs experience")
ax1.set_xlabel("years of experience ")
ax1.set_ylabel("salary")
ax1.scatter(x, y, color="red")
ax1.plot(x_train, x_pridct, color="blue")

# test set result

ax2.set_xlabel("years of experience ")
ax2.set_ylabel("salary")
ax2.scatter(x_test, y_test, color="red")
ax2.plot(x_train, x_pridct, color="blue")

plt.show()
