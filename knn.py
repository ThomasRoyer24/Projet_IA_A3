import pandas as pd
import numpy as np
import  math

def different_or_not(x,y):
  return int(x==y)

def euclidean_distance(x1, x2):
    result = math.sqrt((x1['latitude']-x2["latitude"])**2 + (x1['longitude']-x2["longitude"])**2) + different_or_not(x1["descr_athmo"],x2["descr_athmo"]) + different_or_not(x1["descr_lum"],x2["descr_lum"]) + different_or_not(x1["descr_etat_surf"],x2["descr_etat_surf"]) + different_or_not(x1["descr_dispo_secu"],x2["descr_dispo_secu"])
    return result

data = pd.read_csv("final_data.csv")




def knn(data,k):
    score = 0
    X = data.iloc[:, :6]
    Y = data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = X[:60000], X[60000:], Y[:60000], Y[60000:]
    tab_dist = []
    tab_neighbor = []
    sum_y_train = 0
    for i in range (len(X_test)):
        print("Etape : ", i)
        for j in range (len(X_train)):
            tab_dist.append(euclidean_distance(X_test.iloc[i], X_train.iloc[j]))
        for j in range(k):
            tab_neighbor.append(tab_dist.index(min(tab_dist)))
            del tab_dist[tab_dist.index(min(tab_dist))]
        for j in range(k):
            sum_y_train += Y_train.iloc[tab_neighbor[j]]
        y_predict = round(sum_y_train / k)
        if y_predict == Y_test.iloc[i]:
            score+=1
    return (score / len(X_test))*100

score = knn(data,3)
print("Le score de knn : ",score)