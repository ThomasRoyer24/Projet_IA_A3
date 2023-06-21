import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples
import numpy as np



data = pd.read_csv("final_data.csv")

# Assuming the DataFrame is named "data"
coordinates = data[['longitude', 'latitude']]  # Extract longitude and latitude columns

# Convert DataFrame to NumPy array
coordinates_array = coordinates.to_numpy()

k_values = range(2, 15)  # Range of values for k
#
# davies_scores = []
#
# # Iterate over each value of k
# for k in k_values:
#     # Apply K-means algorithm
#     kmeans = KMeans(n_clusters=k, max_iter=5)
#     kmeans.fit(coordinates_array)
#
#     # Retrieve the cluster labels
#     cluster_labels = kmeans.labels_
#
#     # Calculate the Davies-Bouldin score
#     davies_score = davies_bouldin_score(coordinates_array, cluster_labels)
#
#     # Append the score to the list
#     davies_scores.append(davies_score)
#
# # Plot the Davies-Bouldin scores
# plt.plot(k_values, davies_scores, marker='o')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Davies-Bouldin Score')
# plt.title('Davies-Bouldin Score for Different Values of k')
# plt.show()
#

k = 13
# Apply K-means algorithm
kmeans = KMeans(n_clusters=k, max_iter=5)
kmeans.fit(coordinates_array)

# Retrieve the cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
data['cluster'] = cluster_labels

silhouette_avg = silhouette_score(coordinates_array, cluster_labels)
calinski_score = calinski_harabasz_score(coordinates_array, cluster_labels)
davies_bouldin = davies_bouldin_score(coordinates_array, cluster_labels)

print("Silhouette score:", silhouette_avg)
print("Calinski-Harabasz score:", calinski_score)
print("Davies-Bouldin score:", davies_bouldin)


#
# # Define the range of values for k
# k_values = range(2, 14)
#
# # Initialize an empty list to store the Calinski-Harabasz scores
# calinski_scores = []
#
# # Iterate over each value of k
# for k in k_values:
#     # Apply K-means algorithm
#     kmeans = KMeans(n_clusters=k, max_iter=5)
#     kmeans.fit(coordinates_array)
#
#     # Retrieve the cluster labels
#     cluster_labels = kmeans.labels_
#
#     # Calculate the Calinski-Harabasz score
#     calinski_score = calinski_harabasz_score(coordinates_array, cluster_labels)
#
#     # Append the score to the list
#     calinski_scores.append(calinski_score)
#
# # Plot the Calinski-Harabasz scores
# plt.plot(k_values, calinski_scores, marker='o')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Calinski-Harabasz Score')
# plt.title('Calinski-Harabasz Score for Different Values of k')
# plt.show()
silhouette_values = silhouette_samples(coordinates_array, cluster_labels)

# Get the number of unique clusters
unique_labels = np.unique(cluster_labels)
num_clusters = len(unique_labels)

# Create a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Set y-axis limits and initialize the y_ticks variable
y_lower, y_upper = 0, 0
y_ticks = []

# Loop over each cluster
for i, label in enumerate(unique_labels):
    # Get the silhouette scores for data points in the current cluster
    cluster_silhouette_values = silhouette_values[cluster_labels == label]

    # Sort the silhouette scores in ascending order
    cluster_silhouette_values.sort()

    # Take only the first 2000 values
    cluster_silhouette_values = cluster_silhouette_values[:2000]

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
ax.axvline(x=silhouette_avg, color="red", linestyle="--")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_yticks(y_ticks)
ax.set_title("Silhouette Plot")
ax.grid(True)

plt.show()





#
# plt.rcParams["figure.figsize"] = [7.50, 7.50]
# plt.rcParams["figure.autolayout"] = True
#
# tab_color = ['#FF0000',
#              '#FF8800',
#              '#FFFF00',
#              '#FFFF88',
#              '#888888',
#              '#00FF00',
#              '#00FF88',
#              '#88FF00',
#              '#00FFFF',
#              '#0000FF',
#              '#0088FF',
#              '#8800FF',
#              '#8888FF']
#
# for j in range(k):
#   for i in range(len(data)):
#     if (data['cluster'][i] == j):
#       plt.plot( data['longitude'][i], data['latitude'][i], color=tab_color[j], marker='.', linestyle='')
#   plt.plot(kmeans.cluster_centers_[j][0], kmeans.cluster_centers_[j][1], color="000000", marker='o', linestyle='')
#   print("itr :", j)
#
#
# plt.show()
#
#
