import pandas as pd

data = pd.read_csv("/content/drive/MyDrive/TP_IA/projet_IA/prepared_data.csv", index_col=0, header=0)



data_red = data.drop(columns=["num_veh", "id_usa", "an_nais", "descr_motif_traj", "ville", "date", "id_code_insee", "descr_cat_veh", "descr_agglo", "description_intersection", "age", "place", "descr_type_col", "region"])

#pourcentage de perte :
#(14/21)*100 = 66.66%

