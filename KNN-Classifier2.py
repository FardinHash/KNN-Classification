#libraries

import os
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data= pd.read_csv('/content/Iris.csv')

#feature selection

feature_columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X= data[feature_columns].values
y= data['Species'].values

#spliting data

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 0)

#model create

model= KNeighborsClassifier(n_neighbors=7)

#fit the model with train test

model.fit(X_train, y_train)

#prediction
pred= model.predict(X_test)

#confusion matrix and accuracy
conf_mat= confusion_matrix(y_test, pred)
acc_score= accuracy_score(y_test, pred)

print(acc_score)
