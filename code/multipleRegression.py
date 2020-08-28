from numpy import array
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataframe = pd.read_csv("./DataSets/50_Startups.csv")

# fetch data
profit = dataframe.iloc[:, -1].values
otherdate = dataframe.iloc[:, :-1].values

# one hot encoding
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])],
                       remainder="passthrough")

otherdate = array(ct.fit_transform(otherdate))

(otherdata_train, otherdate_test, profit_train,
 profit_test) = train_test_split(otherdate,
                                 profit,
                                 test_size=0.2,
                                 shuffle=False)

regressor = LinearRegression()
regressor.fit(otherdata_train, profit_train)

profit_pridct = regressor.predict(otherdate_test)
np.set_printoptions(precision=2)
print(
    np.concatenate(
        (
            profit_pridct.reshape(len(profit_pridct), 1),
            profit_test.reshape(len(profit_test), 1),
        ),
        axis=1,
    ))
