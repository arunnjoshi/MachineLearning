from numpy.lib.polynomial import poly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# preprocessing csv file
dataset = pd.read_csv("./classification/datasets/social_network_ads.csv")

purchased = dataset.iloc[:, -1].values
other_data = dataset.iloc[:, :-1].values
(other_data_train, other_data_test, purchased_train, purchased_test) = train_test_split(
    other_data, purchased, random_state=0, test_size=0.25
)

# feature scaling
sc = StandardScaler()
other_data_train = sc.fit_transform(other_data_train)
other_data_test = sc.fit_transform(other_data_test)

# train model
regressor = SVC(kernel="rbf")
regressor.fit(other_data_train, purchased_train)

value = sc.transform([[32, 150000]])
# print(regressor.predict(value))
result = regressor.predict(other_data_test)
# print predict and real values
print(
    np.concatenate(
        (
            purchased_test.reshape(len(purchased_test), 1),
            result.reshape(len(result), 1),
        ),
        axis=1,
    )
)


# confusion matrix
cm = confusion_matrix(purchased_test, result)
acc = accuracy_score(purchased_test, result)
print(cm, acc)
