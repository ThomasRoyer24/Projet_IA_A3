import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, LeaveOneOut
from sklearn.utils import resample
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier



def prepare_data():
    data = pd.read_csv("final_data.csv")

    X = data.drop('descr_grav', axis=1)
    y = data['descr_grav']

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        locals()["X_train_holdout_" + str(i+1)] = X_train
        locals()["y_train_holdout_" + str(i+1)] = y_train
        locals()["X_test_holdout_" + str(i+1)] = X_test
        locals()["y_test_holdout_" + str(i+1)] = y_test


#leave-one-out
def leave-one-out(:)
    classes = y.unique()
    X_train_selected = pd.DataFrame()  # Stockage des échantillons sélectionnés
    y_train_selected = pd.Series()

    for c in classes:
        class_samples = X[y == c]
        class_samples = resample(class_samples, n_samples=int(0.5 * len(class_samples)), random_state=42)
        X_train_selected = pd.concat([X_train_selected, class_samples])
        y_train_selected = pd.concat([y_train_selected, pd.Series([c] * len(class_samples))])


    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X_train_selected):
        X_train_LeaveOneOut, X_test_LeaveOneOut = X_train_selected.iloc[train_index], X_train_selected.iloc[test_index]
        y_train_LeaveOneOut, y_test_LeaveOneOut = y_train_selected.iloc[train_index], y_train_selected.iloc[test_index]


#Knn
def knn()
    for i in range(5):
      print("Précision du modèle :",str(i+1))
      for t in range(15,30):
        knn = KNeighborsClassifier(n_neighbors=t)

        # Entraînement du modèle
        knn.fit(locals()["X_train_holdout_" + str(i+1)], locals()["y_train_holdout_" + str(i+1)])

        # Prédiction sur les données de test
        y_pred = knn.predict(locals()["X_test_holdout_" + str(i+1)])

        # Évaluation de la précision du modèle
        accuracy = accuracy_score(locals()["y_test_holdout_" + str(i+1)], y_pred)

        print(accuracy * 100)

#random_forest
def random_forest_grid()
    random_forest = RandomForestClassifier()

    # Définir la grille des paramètres à tester
    param_grid = {
        'n_estimators': [50, 100, 150,200,250,300,350,400],  # Nombre d'arbres de décision dans la forêt
        'max_depth': [5,10,15,20,25,30,35,40],       # Profondeur maximale de chaque arbre
    }

    # Créer une instance de GridSearchCV avec le modèle et la grille des paramètres
    grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5)

    # Effectuer la recherche par grille sur les données d'entraînement
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres trouvés
    print("Meilleurs paramètres :", grid_search.best_params_)

def random_forest():
    random_forest = RandomForestClassifier(max_depth=15,n_estimators=150)

    accuracy_tree=0
    for i in range(5):

      random_forest.fit(locals()["X_train_holdout_" + str(i+1)], locals()["y_train_holdout_" + str(i+1)])

      #score d’échantillons bien classifiés sur le jeu de données de test
      accuracy_tree += random_forest.score(locals()["X_test_holdout_" + str(i+1)], locals()["y_test_holdout_" + str(i+1)])

    accuracy_tree /=5
    print(accuracy_tree)

#Support Vector Machine

def SVM():
    # Créer une instance de la classe SVC (Support Vector Classifier)
    svm = SVC(C=1,gamma=1,kernel='rbf')
    accuracy_svm = 0
    # Entraîner le modèle sur les données d'entraînement
    for i in range(5):
        svm.fit(locals()["X_train_holdout_" + str(i+1)], locals()["y_train_holdout_" + str(i+1)])

        y_pred = svm.predict(locals()["X_test_holdout_" + str(i+1)])

        # Calculer la précision du modèle
        accuracy_svm += accuracy_score(locals()["y_test_holdout_" + str(i+1)], y_pred)

    accuracy_svm /=5
    print(accuracy_svm)

def MLP():
    accuracy_MLP = 0
    # Créer une instance de la classe MLPClassifier (Multilayer Perceptron Classifier)
    mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100, 100), activation='logistic', alpha=0.0001)

    for i in range(5):
        mlp.fit(locals()["X_train_holdout_" + str(i+1)], locals()["y_train_holdout_" + str(i+1)])

        y_pred = mlp.predict(locals()["X_test_holdout_" + str(i+1)])

        accuracy_MLP += accuracy_score(locals()["y_test_holdout_" + str(i+1)], y_pred)

    accuracy_MLP /=5
    print(accuracy_MLP)