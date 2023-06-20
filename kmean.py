import pandas as pd
import math

data = pd.read_csv("final_data.csv")


def kmean(distance, k, n):
  centro_lat = []
  centro_long = []
  tab_cluster = [None] * len(data)
  temp = 99999999
  for i in range (k):
    centro_lat.append(data["latitude"][i])
    centro_long.append(data["longitude"][i])
  for i in range(len(data)):
    for j in range (len(centro_lat)):
      if (distance == "L1"):
        if (dist_l1(centro_long[j],centro_lat[j],data["longitude"][i],data["latitude"][i]) <= temp):
          temp = dist_l1(centro_long[j],centro_lat[j],data["longitude"][i],data["latitude"][i])
          tab_cluster[i] = j
    temp = 999999
  for i in range(k):
    if i in tab_cluster:
      res_centroide = mean_cluster(tab_cluster, i)
      centro_lat[i] = res_centroide[0]
      centro_long[i] = res_centroide[1]
  if n == 0:
    final = [tab_cluster, centro_lat, centro_long]
    print(tab_cluster)
    print(centro_lat)
    print(centro_long)
    return final
  print("itération : ", n)
  kmean(distance,k,n-1)

  # print(tab_cluster)

def mean_cluster(tab, cluster):
  temp_long = 0
  temp_lat = 0
  number = 0
  for i in range(len(data)):
    if tab[i] == cluster:
      temp_long += data["longitude"][i]
      temp_lat += data["latitude"][i]
      number+=1
    if cluster == 1:
      print(tab[i])
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


kmean("L1",1,100)