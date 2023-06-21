import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.basemap import Basemap



data = pd.read_csv("final_data.csv")

coordinates = data[['longitude', 'latitude']]  # Extract longitude and latitude columns
coordinates_array = coordinates.to_numpy()
k = 3
kmeans = KMeans(n_clusters=13, max_iter=100)
kmeans.fit(coordinates_array)
cluster_labels = kmeans.labels_
data['cluster'] = cluster_labels

# Create a new figure
fig = plt.figure(figsize=(10, 10))

# Create a Basemap instance for France
m = Basemap(llcrnrlon=-5, llcrnrlat=40, urcrnrlon=10, urcrnrlat=52, resolution='i')

# Draw coastlines, countries, and fill the continents
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.fillcontinents(color='lightgray')

# Plot the data points on top of the map
m.scatter(data['longitude'], data['latitude'], c=data['cluster'], latlon=True)

# Plot the cluster centers as markers
for center in kmeans.cluster_centers_:
    x, y = m(center[0], center[1])
    plt.plot(x, y, color="#000000", marker='o', linestyle='')

# Show the plot
plt.show()
# california_img = mpimg.imread("France-region.png")
# data.plot(x="longitude", y="latitude", alpha=0.4, colormap='gist_rainbow', figsize=(10,7))
# plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5, cmap=plt.get_cmap("jet"))
# plt.legend()



