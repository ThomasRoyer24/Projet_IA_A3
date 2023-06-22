import sys
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ast
import json


#
# data = pd.read_csv("final_data.csv")
# X = data.iloc[:, :6]
# y = data.iloc[:, -1]
#
# bestac = 0
# bestt = 0
#
# for i in range(5):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     locals()["X_train_holdout_" + str(i+1)] = X_train
#     locals()["y_train_holdout_" + str(i+1)] = y_train
#     locals()["X_test_holdout_" + str(i+1)] = X_test
#     locals()["y_test_holdout_" + str(i+1)] = y_test
#
# for i in range(5):
#   print("Précision du modèle :",str(i+1))
#   for t in range(15,30):
#     knn = KNeighborsClassifier(n_neighbors=t)
#
#     # Entraînement du modèle
#     knn.fit(locals()["X_train_holdout_" + str(i+1)], locals()["y_train_holdout_" + str(i+1)])
#
#     # Prédiction sur les données de test
#     y_pred = knn.predict(locals()["X_test_holdout_" + str(i+1)])
#
#     # Évaluation de la précision du modèle
#     accuracy = accuracy_score(locals()["y_test_holdout_" + str(i+1)], y_pred)
#
#     if accuracy >bestac:
#         bestac = accuracy
#         bestt = t
#
#     print(accuracy * 100)
#
# print("le meilleur k :", bestt)

def pred_knn(data, accident):
    data = pd.read_csv(data)
    X = data.iloc[:, :6]
    y = data.iloc[:, -1]

    feature_names = ['latitude', 'longitude', 'descr_athmo', 'descr_lum', 'descr_etat_surf', 'descr_dispo_secu']
    X.columns = feature_names

    neigh = KNeighborsClassifier(n_neighbors=26)
    neigh.fit(X, y)
    return neigh.predict(ast.literal_eval(accident))  #[[48.1667,-3.78333,2.0,1.0,3.0,5.0]] exemple

grav = pred_knn(sys.argv[1], sys.argv[2]).tolist()
res = {
    "gravite": grav[0]
}

json_object = json.dumps(res, indent=4)

with open("pred_knn.json", "w") as outfile:
    outfile.write(json_object)

print(res)