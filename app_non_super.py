import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import csv

data = pd.read_csv("prepared_data.csv", index_col=0, header=0)



data_red = data.drop(columns=["num_veh", "id_usa", "an_nais", "descr_motif_traj", "ville", "date", "id_code_insee", "descr_cat_veh", "descr_agglo", "description_intersection", "age", "place", "descr_type_col", "region"])


#pourcentage de perte :
#(14/21)*100 = 66.66%

#conversion de char en int

#descr_athmo
encoder = OrdinalEncoder()
atmo = data_red[["descr_athmo"]]
data_red[["descr_athmo"]] = encoder.fit_transform(atmo)

dict_athmo = {'Normal':3, 'Pluie légère':5, 'Temps couvert':6, 'Pluie forte':4, 'Temps éblouissant':7, 'Neige – grêle':2, 'Autre':0, 'Brouillard – fumée':1, 'Vent fort – tempête':8}
dict_athmo = {v: k for k, v in dict_athmo.items()}

#descr_lum
encoder = OrdinalEncoder()
lum = data_red[["descr_lum"]]
data_red[["descr_lum"]] = encoder.fit_transform(lum)

dict_lum = {0:'Crépuscule ou aube', 1:'Nuit avec éclairage public allumé', 2:'Nuit avec éclairage public non allumé', 3:'Nuit sans éclairage public', 4:'Plein jour'}

#descr_surf
encoder = OrdinalEncoder()
surf = data_red[["descr_etat_surf"]]
data_red[["descr_etat_surf"]] = encoder.fit_transform(surf)

dict_surf = {7:'Normale', 6:'Mouillée', 8:'Verglacée', 0:'Autre', 3:'Enneigée', 2:'Corps gras – huile', 4:'Flaques', 1:'Boue', 5:'Inondée'}

#descr_secu
encoder = OrdinalEncoder()
secu = data_red[["descr_dispo_secu"]]
data_red[["descr_dispo_secu"]] = encoder.fit_transform(secu)

#print(data["descr_dispo_secu"].value_counts())
#print(data_red["descr_dispo_secu"].value_counts())

dict_secu = {14:"Utilisation d'une ceinture de sécurité",
             11:"Utilisation d'un casque",
             7:"Présence d'une ceinture de sécurité - Utilisation non déterminable",
             0:"Autre - Non déterminable",
             3:"Présence d'un casque - Utilisation non déterminable",
             8:"Présence de ceinture de sécurité non utilisée",
             4:"Présence d'un casque non utilisé",
             12:"Utilisation d'un dispositif enfant",
             2:"Autre - Utilisé",
             1:"Autre - Non utilisé",
             13:"Utilisation d'un équipement réfléchissant",
             10:"Présence équipement réfléchissant - Utilisation non déterminable",
             6:"Présence d'un équipement réfléchissant non utilisé",
             9:"Présence dispositif enfant - Utilisation non déterminable",
             5:"Présence d'un dispositif enfant non utilisé"}

data_red.to_csv("final_data.csv", index=False)