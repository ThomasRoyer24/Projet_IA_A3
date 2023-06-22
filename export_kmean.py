import json
# !pip install simplejson
# import simplejson
import sys


def kmeans_fct(Latitude_int, Longitude_int, path_fic_centroide):
    kmeans = KMeans(n_clusters=13, max_iter=300)
    clusters_centers = np.loadtxt(path_fic_centroide)
    kmeans.fit(clusters_centers)
    kmeans.cluster_centers_ = clusters_centers
    return kmeans.predict([[Latitude_int, Longitude_int]])[0]


# print(kmeans_fct(45,6, '/content/drive/MyDrive/TP_IA/projet_IA/clusters.txt'))

result = kmeans(sys.argv[1], sys.argv[2], sys.argv[3])

print(result)

# with open('data.json', 'w') as mon_fichier:
# 	json.dump(kmeans_fct, mon_fichier)
