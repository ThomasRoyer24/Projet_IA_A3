import pandas as pd
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split



data = pd.read_csv("final_data.csv")

X = data.drop('descr_grav', axis=1)
y = data['descr_grav']

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    locals()["X_train_holdout_" + str(i+1)] = X_train
    locals()["y_train_holdout_" + str(i+1)] = y_train
    locals()["X_test_holdout_" + str(i+1)] = X_test
    locals()["y_test_holdout_" + str(i+1)] = y_test


#Chercher les meilleurs paramètres à trouvés
def random_forest_grid() :
    random_forest = RandomForestClassifier()

    param_grid = {
        'n_estimators': [50, 100, 150,200,250,300],  # Nombre d'arbres de décision dans la forêt
        'max_depth': [5,10,15,20,25],       # Profondeur maximale de chaque arbre
    }

    # Créer une instance de GridSearchCV avec le modèle et la grille des paramètres
    grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=3)

    # Effectuer la recherche par grille sur les données d'entraînement
    grid_search.fit(X_train, y_train)

    print("Meilleurs paramètres :", grid_search.best_params_)

    # Enregistrer les meilleurs paramètre
    dump(grid_search, 'random_forest.joblib')


#Affiche
def show_grid():
    # Recuprer le fichier
    clf_tree = load('random_forest.joblib')

    # Extraire les données des résultats
    max_depths = np.unique(clf_tree.cv_results_['param_max_depth'].data)
    n_estimators = np.unique(clf_tree.cv_results_['param_n_estimators'].data)
    mean_scores = clf_tree.cv_results_['mean_test_score']
    # Convertir les scores en une matrice 2D
    mean_scores = mean_scores.reshape(len(max_depths), len(n_estimators))

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_scores, interpolation='nearest', cmap='hot')
    plt.title('Optimization of max_depth and n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.colorbar(label='Mean Test Score')
    plt.xticks(np.arange(len(n_estimators)), n_estimators, rotation=45)
    plt.yticks(np.arange(len(max_depths)), max_depths)

    # Ajouter du texte au milieu de chaque case
    for (j, i), label in np.ndenumerate(mean_scores):
        plt.text(i, j, round(label, 2), ha='center', va='center', color='black')

    plt.show()


random_forest = load('random_forest.joblib')
print(random_forest.best_estimator_)
accuracy_tree=0
cm =0
precision =0
recall=0
for i in range(5):

    y_pred = random_forest.predict(locals()["X_test_holdout_" + str(i+1)])

    accuracy_tree += random_forest.score(locals()["X_test_holdout_" + str(i + 1)],
                                         locals()["y_test_holdout_" + str(i + 1)])

    cm_test = confusion_matrix(locals()["y_test_holdout_" + str(i+1)], y_pred)

    # Précision
    precision_test = precision_score(locals()["y_test_holdout_" + str(i+1)], y_pred,average='weighted')

    # Rappel
    recall_test = recall_score(locals()["y_test_holdout_" + str(i+1)], y_pred,average='weighted')


    cm += cm_test
    precision += precision_test
    recall += recall_test

accuracy_tree /=5
recall /=5
precision /=5
for i in range(len(cm)):
    for j in range(len(cm[i])):
        cm[i][j] /= 5

print("Accuracy",accuracy_tree)
print("Rappel :", recall)
print("Précision :", precision)
print("Matrice de confusion :\n", cm)
