__author__ = 'LobaAjisafe'

import numpy as np
import pandas as pd


dataset = pd.read_csv('computer_survey.csv')
Xo = dataset.iloc[:, [1, 2, 3, 4, 5, 7]].values
yo = dataset.iloc[:, -2].values
X = dataset.iloc[:, [1, 2, 3, 4, 5, 7]].values
y = dataset.iloc[:, -2].values


# df1 = pd.DataFrame(['15-20', 'M', 'USA', 'Android', 5, ])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_0.fit_transform(Xo[:, 0])
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(Xo[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(Xo[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(Xo[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, -1] = labelencoder_X_4.fit_transform(Xo[:, -1])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(yo)
onehotencoder = OneHotEncoder(categorical_features=[0, 3])
X = onehotencoder.fit_transform(X).toarray()

def encode():
    global X
    global y
    X[:, 0] = labelencoder_X_0.fit_transform(Xo[:, 0])
    X[:, 1] = labelencoder_X_1.fit_transform(Xo[:, 1])
    X[:, 2] = labelencoder_X_2.fit_transform(Xo[:, 2])
    X[:, 3] = labelencoder_X_3.fit_transform(Xo[:, 3])
    X[:, -1] = labelencoder_X_4.fit_transform(Xo[:, -1])
    y = labelencoder_y.fit_transform(yo)

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
from keras.layers import Dense, Dropout

n = len(X[0])
classifier = Sequential()
classifier.add(Dense(output_dim=((n+1)//2), init='uniform', activation='relu', input_dim=n))
classifier.add(Dropout(p=.1))
classifier.add(Dense(output_dim=((n+1)//2), init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# classifier.fit(X_train, y_train, batch_size=1, nb_epoch=10)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# y_prediction = []
# for i in range(len(y_pred)):
#     if y_pred[i] > .5:
#         y_prediction.append('Windows')
#     if y_pred[i] < .5:
#         y_prediction.append('Mac')
# y_prediction = np.array(y_prediction)
# y_pred = (y_pred > .5)
#
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV


def build_classifier(optimizer, activation, p, f_activation):
    n = len(X[0])
    classify = Sequential()
    classify.add(Dense(output_dim=((n+1)//2), init='uniform', activation=activation, input_dim=n))
    classify.add(Dropout(p=p))
    classify.add(Dense(output_dim=((n+1)//2), init='uniform', activation=activation))
    classify.add(Dense(output_dim=1, init='uniform', activation=f_activation))
    classify.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classify


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
    'batch_size': [1],
    'epochs': [1],
    'optimizer': ['adam'],
    'activation': ['relu'],
    'p': [0],
    'f_activation': ['sigmoid']
}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)

# if __name__ == '__main__':
#     accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
#     print(accuracies)
#     print(accuracies.mean())
#     print(accuracies.std())
