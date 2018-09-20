__author__ = 'LobaAjisafe'

import numpy as np
import pandas as pd


dataset = pd.read_csv('computer_survey.csv')
# Xo = dataset.iloc[:, [1, 2, 3, 4, 5, 7]].values
# yo = dataset.iloc[:, -2].values
X = dataset.iloc[:, [1, 2, 3, 4, 5, 7]].values
y = dataset.iloc[:, -2].values

# df1 = pd.DataFrame(['15-20', 'M', 'USA', 'Android', 5, ])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def encode():
    global X
    global y
    labelencoder_X_0 = LabelEncoder()
    X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    labelencoder_X_3 = LabelEncoder()
    X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
    labelencoder_X_4 = LabelEncoder()
    X[:, -1] = labelencoder_X_4.fit_transform(X[:, -1])
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)
    onehotencoder = OneHotEncoder(categorical_features=[0, 3, 4])
    X = onehotencoder.fit_transform(X).toarray()


encode()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    n = len(X[0])
    classify = Sequential()
    classify.add(Dense(output_dim=((n+1)//2), init='uniform', activation='relu', input_dim=n))
    classify.add(Dense(output_dim=((n+1)//2), init='uniform', activation='relu'))
    classify.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classify.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classify


classifier = KerasClassifier(build_fn=build_classifier(), batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
