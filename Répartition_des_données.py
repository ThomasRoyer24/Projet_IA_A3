import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



data = pd.read_csv("final_data.csv")

X = data.drop('descr_grav', axis=1)
y = data['descr_grav']

#holdout
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    locals()["X_train_holdout_" + str(i+1)] = X_train
    locals()["y_train_holdout_" + str(i+1)] = y_train
    locals()["X_test_holdout_" + str(i+1)] = X_test
    locals()["y_test_holdout_" + str(i+1)] = y_test


#leave-one-out -- KNN
classes = y.unique()
X_train_selected = pd.DataFrame()
y_train_selected = pd.Series()

for c in classes:
    class_samples = X[y == c]
    class_samples = resample(class_samples, n_samples=int(0.1 * len(class_samples)), random_state=42)
    X_train_selected = pd.concat([X_train_selected, class_samples])
    y_train_selected = pd.concat([y_train_selected, pd.Series([c] * len(class_samples))])

loo = LeaveOneOut() #0.5357636012151339

accuracy = 0
g =0
for train_index, test_index in loo.split(X_train_selected):
    X_train_LeaveOneOut, X_test_LeaveOneOut = X_train_selected.iloc[train_index], X_train_selected.iloc[test_index]
    y_train_LeaveOneOut, y_test_LeaveOneOut = y_train_selected.iloc[train_index], y_train_selected.iloc[test_index]
    for i in range(5):
      for t in range(29,30):
        knn = KNeighborsClassifier(n_neighbors=t)

        knn.fit(X_train_LeaveOneOut, y_train_LeaveOneOut)

        y_pred = knn.predict(X_test_LeaveOneOut)

        accuracy += accuracy_score(y_test_LeaveOneOut, y_pred)
        g+=1

#moyenne
accuracy /= g

print(accuracy)

#holdout -- KNN
accuracy = 0
g =0
for i in range(5):

  for t in range(29,30):
    knn = KNeighborsClassifier(n_neighbors=t)

    knn.fit(locals()["X_train_holdout_" + str(i+1)], locals()["y_train_holdout_" + str(i+1)])

    y_pred = knn.predict(locals()["X_test_holdout_" + str(i+1)])

    accuracy += accuracy_score(locals()["y_test_holdout_" + str(i+1)], y_pred)
    g+=1

#moyenne
accuracy /= g

print(accuracy)