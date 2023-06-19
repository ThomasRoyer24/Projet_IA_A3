import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, LeaveOneOut
from sklearn.utils import resample

data = pd.read_csv("prepared_data.csv", index_col=0, header=0)

X = data.drop('descr_grav', axis=1)
y = data['descr_grav']
holdout_repeats = 5
test_size = 0.2

for i in range(holdout_repeats):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    locals()["X_train_" + str(i + 1)] = X_train
    locals()["y_train_" + str(i + 1)] = y_train
    locals()["X_test_" + str(i + 1)] = X_test
    locals()["y_test_" + str(i + 1)] = y_test


classes = y.unique()
X_train_selected = pd.DataFrame()  # Stockage des échantillons sélectionnés
y_train_selected = pd.Series()

for c in classes:
    class_samples = X[y == c]
    class_samples = resample(class_samples, n_samples=int(0.1 * len(class_samples)), random_state=42)
    X_train_selected = pd.concat([X_train_selected, class_samples])
    y_train_selected = pd.concat([y_train_selected, pd.Series([c] * len(class_samples))])

