import json
import sys


#Création de la fonction kmean demandé
#Prend la latitude, la longitude du possible accident et le chemin du fichier contenant les centroïdes
def kmeans_fct(Latitude, Longitude, path_fic_centroide):
    kmeans = KMeans(n_clusters=13, max_iter=300)
    clusters_centers = np.loadtxt(path_fic_centroide)
    kmeans.fit(clusters_centers)
    kmeans.cluster_centers_ = clusters_centers
    return kmeans.predict([[Latitude, Longitude]])[0]

result = kmeans(sys.argv[1], sys.argv[2], sys.argv[3])

print(result)

# with open('data.json', 'w') as mon_fichier:
# 	json.dump(kmeans_fct, mon_fichier)
