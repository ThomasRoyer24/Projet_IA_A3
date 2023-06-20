import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("final_data.csv")

# Assuming the DataFrame is named "data"
coordinates = data[['longitude', 'latitude']]  # Extract longitude and latitude columns

# Convert DataFrame to NumPy array
coordinates_array = coordinates.to_numpy()

# Specify the number of clusters (k)
k = 3

# Apply K-means algorithm
kmeans = KMeans(n_clusters=1, max_iter=100)
kmeans.fit(coordinates_array)

# Retrieve the cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
data['cluster'] = cluster_labels

plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True

tab_color = ['#FF0000',
             '#FF8800',
             '#FFFF00',
             '#FFFF88',
             '#888888',
             '#00FF00',
             '#00FF88',
             '#88FF00',
             '#00FFFF',
             '#0000FF',
             '#0088FF',
             '#8800FF',
             '#8888FF']

for j in range(13):
  for i in range(len(data)):
    if (data['cluster'][i] == j):
      plt.plot( data['longitude'][i], data['latitude'][i], color=tab_color[j], marker='.', linestyle='')
  plt.plot(kmeans.cluster_centers_[j][0], kmeans.cluster_centers_[j][1], color="000000", marker='o', linestyle='')
  print("itr :", j)


plt.show()


