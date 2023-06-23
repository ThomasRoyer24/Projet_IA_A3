import pandas as pd
import math
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import random
from sklearn.metrics import calinski_harabasz_score



data = pd.read_csv("final_data.csv")


def kmean(distance, k, n):
  centro_lat = []
  centro_long = []
  tab_cluster = [None] * len(data)
  #initialisation de notre valeur tempo tres grande pour les comparaisons
  temp = 99999999
  #Recuperation de nos centroïdes
  for i in range (k):
    centro_lat.append(random.choice(data["latitude"]))
    centro_long.append(random.choice(data["longitude"]))
  #Association de chaque point a un cluster par rapport aux centroïdes
  for i in range(len(data)):
    for j in range (len(centro_lat)):
      if (distance == "L1"):
        if (dist_l1(centro_long[j],centro_lat[j],data["longitude"][i],data["latitude"][i]) <= temp):
          temp = dist_l1(centro_long[j],centro_lat[j],data["longitude"][i],data["latitude"][i])
          tab_cluster[i] = j
      if (distance == "L2"):
        if (dist_l2(centro_long[j],centro_lat[j],data["longitude"][i],data["latitude"][i]) <= temp):
          temp = dist_l2(centro_long[j],centro_lat[j],data["longitude"][i],data["latitude"][i])
          tab_cluster[i] = j
      if (distance == "haversine"):
        if (dist_haversine(centro_long[j],centro_lat[j],data["longitude"][i],data["latitude"][i]) <= temp):
          temp = dist_haversine(centro_long[j],centro_lat[j],data["longitude"][i],data["latitude"][i])
          tab_cluster[i] = j
    temp = 999999
  #Test si nous sommes a la dernière itération
  if n == 0:
    final = [tab_cluster, centro_lat, centro_long]
    tab_cluster_array = np.array(tab_cluster[:2000])
    silhouette = silhouette_score(data[['longitude', 'latitude']].head(2000), tab_cluster_array)
    print("Silhouette score:", silhouette)
    silhouette_values = silhouette_samples(data[['longitude', 'latitude']].head(2000), tab_cluster_array)

    # Get the number of unique clusters
    unique_labels = np.unique(tab_cluster_array)
    num_clusters = len(unique_labels)

    # Create a subplot with 1 row and 1 column
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Set y-axis limits and initialize the y_ticks variable
    y_lower, y_upper = 0, 0
    y_ticks = []

    # Loop over each cluster
    for i, label in enumerate(unique_labels):
      # Get the silhouette scores for data points in the current cluster
      cluster_silhouette_values = silhouette_values[tab_cluster_array == label]

      # Sort the silhouette scores in ascending order
      cluster_silhouette_values.sort()

      # Calculate the size of the current cluster
      cluster_size = cluster_silhouette_values.shape[0]

      # Update y-axis limits and y_ticks
      y_upper += cluster_size
      y_ticks.append((y_lower + y_upper) / 2)

      # Color the silhouette plot for the current cluster
      color = plt.cm.get_cmap("Spectral")(i / num_clusters)
      ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, alpha=0.7)

      # Update y_lower for the next cluster
      y_lower += cluster_size

    # Set the properties of the silhouette plot
    ax.axvline(x=silhouette_score(data[['longitude', 'latitude']].head(2000), tab_cluster_array), color="red", linestyle="--")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks(y_ticks)
    ax.set_title("Silhouette Plot")
    ax.grid(True)

    plt.show()
    return final
  #Préparation pour la prochgaine itération, calcule des nouveaux centroïdes
  for i in range(k):
    if i in tab_cluster:
      res_centroide = mean_cluster(tab_cluster, i)
      centro_lat[i] = res_centroide[0]
      centro_long[i] = res_centroide[1]
  print("itération : ", n)
  return kmean(distance,k,n-1) #Récursivitées n fois


def davies(k):
  davies_scores = []
  for i in range(2, k + 1):
    al = kmean("L1",i,3)
    tab_cluster = al[0]
    davies_score = davies_bouldin_score(data[['longitude', 'latitude']], tab_cluster)
    davies_scores.append(davies_score)

  # Plot the Davies-Bouldin scores
  plt.plot(range(2, k + 1), davies_scores, marker='o')
  plt.xlabel('Number of clusters (k)')
  plt.ylabel('Davies-Bouldin Score')
  plt.title('Davies-Bouldin Score for Different Values of k')
  plt.show()

def kalinski(k):
  kalinski_scores = []
  for i in range(2, k + 1):
    print("TEST : ",i)
    al = kmean("L1",i,3)
    tab_cluster = al[0]
    kalinski_score = calinski_harabasz_score(data[['longitude', 'latitude']], tab_cluster)
    kalinski_scores.append(kalinski_score)

  # Plot the Davies-Bouldin scores
  plt.plot(range(2, k + 1), kalinski_scores, marker='o')
  plt.xlabel('Number of clusters (k)')
  plt.ylabel('Calinski-Harabasz Score')
  plt.title('Calinski-Harabasz Score for Different Values of k')
  plt.show()


def mean_cluster(tab, cluster):
  temp_long = 0
  temp_lat = 0
  number = 0
  for i in range(len(data)):
    if tab[i] == cluster:
      temp_long += data["longitude"][i]
      temp_lat += data["latitude"][i]
      number+=1
  # print(cluster)
  # print(number)
  temp_long = temp_long /number
  temp_lat = temp_lat / number
  tab = [temp_lat,temp_long]
  return tab

def dist_l1(long1, lat1, long2, lat2):
  return (abs(long2 - long1) + abs(lat2 - lat1))

def dist_l2(long1, lat1, long2, lat2):
  return math.sqrt(((long2 - long1) * (long2 - long1)) + ((lat2 - lat1) * (lat2 - lat1)))

def dist_haversine(long1, lat1, long2, lat2):
  return 2*math.asin((math.sqrt((math.sin((math.radians(lat1)-math.radians(long1))/2)*math.sin((math.radians(lat1)-math.radians(long1))/2)+math.cos(math.radians(lat1))*math.cos(math.radians(long1))*math.sin((math.radians(lat2)-math.radians(long2))/2))*math.sin((math.radians(lat2)-math.radians(long2))/2))))


tab_km = kmean("L1",13,100)
data['cluster'] = tab_km[0]

#PARTIE VISUALISATION :

pip install basemap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = [7.50, 7.50]
# plt.rcParams["figure.autolayout"] = True

# Create a new figure
fig = plt.figure(figsize=(10, 15))

# Create a Basemap instance for France
m = Basemap(llcrnrlon=-5, llcrnrlat=40, urcrnrlon=10, urcrnrlat=52, resolution='i')

# Draw coastlines, countries, and fill the continents
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.fillcontinents(color='lightgray')

# Plot the data points on top of the map
m.scatter(data['longitude'], data['latitude'], c=data['cluster'], latlon=True)

# Plot the cluster centers as markers
for i in range(len(tab_km[1])):
  x, y = m(tab_km[2][i], tab_km[1][i])
  plt.plot(x, y, color="#000000", marker='o', linestyle='')

plt.show()