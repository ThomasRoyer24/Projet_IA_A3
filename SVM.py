import pandas as pd
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



data = pd.read_csv("final_data.csv")

X = data.drop('descr_grav', axis=1)
y = data['descr_grav']

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    locals()["X_train_holdout_" + str(i+1)] = X_train
    locals()["y_train_holdout_" + str(i+1)] = y_train
    locals()["X_test_holdout_" + str(i+1)] = X_test
    locals()["y_test_holdout_" + str(i+1)] = y_test

def SVM_grid():
    svm = SVC()

    # Définir la grille des paramètres à tester
    param_grid = {
        'C': [0.1, 1, 0.01,10],  # Paramètre de régularisation C
        'kernel': ['sigmoid', 'rbf'],  # Type de noyau
        'gamma': [1,'scale', 'auto']  # Coefficient du noyau (uniquement pour les noyaux 'rbf', 'poly', 'sigmoid')
    }

    # Créer une instance de GridSearchCV avec le modèle et la grille des paramètres
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=2,n_jobs=10)

    grid_search.fit(X_train, y_train)

    print("Meilleurs paramètres :", grid_search.best_params_)

    # Enregistrer les meilleurs paramètre
    dump(grid_search, 'SVM.joblib')



#Affiche
def show_grid():
    # Recuprer le fichier
    clf_SVM = load('SVM.joblib')

    # Extraire les données des résultats
    param_C = np.unique(clf_SVM.cv_results_['param_C'].data)
    param_gamma = set(clf_SVM.cv_results_['param_gamma'].data)
    mean_scores = clf_SVM.cv_results_['mean_test_score']
    # fixe des paramètres pour avoir un tableau a 2D et non 3 que on ne peut pas afficher
    mean_scores = mean_scores[clf_SVM.cv_results_["param_kernel"]== "rbf"]
    # Convertir les scores en une matrice 2D
    mean_scores = mean_scores.reshape(len(param_C), len(param_gamma))

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_scores, interpolation='nearest', cmap='hot')
    plt.title('Optimization of C and gamma with kernel = rbf')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar(label='Mean Test Score')
    plt.xticks(np.arange(len(param_gamma)), param_gamma, rotation=45)
    plt.yticks(np.arange(len(param_C)), param_C)

    # Ajouter du texte au milieu de chaque case
    for (j, i), label in np.ndenumerate(mean_scores):
        plt.text(i, j, round(label, 2), ha='center', va='center', color='black')

    plt.show()
show_grid()

SVM = load('SVM.joblib')
print(SVM.best_estimator_)
cm =0
precision =0
recall=0
for i in range(5):


    y_pred = SVM.predict(locals()["X_test_holdout_" + str(i+1)])



    cm_test = confusion_matrix(locals()["y_test_holdout_" + str(i+1)], y_pred)

    # Précision
    precision_test = precision_score(locals()["y_test_holdout_" + str(i+1)], y_pred,average='weighted')

    # Rappel
    recall_test = recall_score(locals()["y_test_holdout_" + str(i+1)], y_pred,average='weighted')

    #accuracy += accuracy_test
    cm += cm_test
    precision += precision_test
    recall += recall_test


recall /=5
precision /=5

for i in range(len(cm)):
    for j in range(len(cm[i])):
        cm[i][j] /= 5

print("Rappel :", recall)
print("Précision :", precision)
print("Matrice de confusion :\n", cm)