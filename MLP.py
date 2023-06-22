import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("final_data.csv")

X = data.drop('descr_grav', axis=1)
y = data['descr_grav']

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    locals()["X_train_holdout_" + str(i+1)] = X_train
    locals()["y_train_holdout_" + str(i+1)] = y_train
    locals()["X_test_holdout_" + str(i+1)] = X_test
    locals()["y_test_holdout_" + str(i+1)] = y_test



def MLP_grid():
    mlp = MLPClassifier()

    # Définir la grille des paramètres à tester
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50),(100,100),(100,100,100)],  # Tailles des couches cachées
        'activation': ['relu', 'tanh','logistic'],  # Fonction d'activation
        'solver': ['adam', 'sgd'],  # Algorithme d'optimisation
        'alpha': [0.0001, 0.001, 0.01]  # Paramètre de régularisation L2
    }

    # Créer une instance de GridSearchCV avec le modèle et la grille des paramètres
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=2,n_jobs=10)

    # Effectuer la recherche par grille sur les données d'entraînement
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs paramètres trouvés
    print("Meilleurs paramètres :", grid_search.best_params_)

    # Enregistrer les meilleurs paramètre
    dump(grid_search, 'MLP.joblib')

def show_grid():
    # Recuprer le fichier
    clf_MLP = load('MLP.joblib')

    # Extraire les données des résultats
    param_alpha = np.unique(clf_MLP.cv_results_['param_alpha'].data)
    param_hidden_layer_sizes = set(clf_MLP.cv_results_['param_hidden_layer_sizes'].data)
    mean_scores = clf_MLP.cv_results_['mean_test_score']


    mean_scores = mean_scores[(clf_MLP.cv_results_["param_activation"] == "tanh") & (clf_MLP.cv_results_["param_solver"] == "adam")]

    # Convertir les scores en une matrice 2D
    mean_scores = mean_scores.reshape(len(param_alpha), len(param_hidden_layer_sizes))

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_scores, interpolation='nearest', cmap='hot')
    plt.title('Optimization of C and gamma with kernel = rbf')
    plt.xlabel('C')
    plt.ylabel('gamma')
    plt.colorbar(label='Mean Test Score')
    plt.xticks(np.arange(len(param_hidden_layer_sizes)), param_hidden_layer_sizes, rotation=45)
    plt.yticks(np.arange(len(param_alpha)), param_alpha)

    # Ajouter du texte au milieu de chaque case
    for (j, i), label in np.ndenumerate(mean_scores):
        plt.text(i, j, round(label, 2), ha='center', va='center', color='black')

    plt.show()

show_grid()
def MLP():
    mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100), activation='tanh', alpha=0.001)

    accuracy=0
    cm =0
    precision =0
    recall=0
    for i in range(5):

        y_pred = MLP.predict(locals()["X_test_holdout_" + str(i+1)])

        accuracy_test = accuracy_score(locals()["X_test_holdout_" + str(i+1)], y_pred)

        cm_test = confusion_matrix(locals()["y_test_holdout_" + str(i+1)], y_pred)

        # Précision
        precision_test = precision_score(locals()["y_test_holdout_" + str(i+1)], y_pred,average='weighted')

        # Rappel
        recall_test = recall_score(locals()["y_test_holdout_" + str(i+1)], y_pred,average='weighted')

        accuracy += accuracy_test
        cm += cm_test
        precision += precision_test
        recall += recall_test


    recall /=5
    precision /=5
    accuracy /=5
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            cm[i][j] /= 5

    print("Rappel :", recall)
    print("Précision :", precision)
    print("Matrice de confusion :\n", cm)