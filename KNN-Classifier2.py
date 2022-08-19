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

model= KNeighborsClassifier(n_neighbors=3)
