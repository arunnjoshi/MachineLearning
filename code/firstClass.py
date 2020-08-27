import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


dataset = pd.read_csv('./datasets/data.csv')
# col selection
colSelection = dataset.iloc[:, -1].values
# row selection
rowSelection = dataset.iloc[:-1, :-1].values

# taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(rowSelection[:, [1, 2]])
rowSelection[:, [1, 2]] = imputer.transform(rowSelection[:, [1, 2]])

# encoding data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                       remainder='passthrough')
rowSelection = np.array(ct.fit_transform(rowSelection))

le = LabelEncoder()
colSelection = le.fit_transform(colSelection)
print(colSelection)