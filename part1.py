import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

data = pd.read_csv("prepared_data.csv")

print(data.info())

print(data["descr_grav"].value_counts())

data.hist(bins=90, figsize=(20,15))
plt.show()

tab =[]

for i in range(len(data)):
    datetime_obj = datetime.strptime(data["date"][i], "%Y-%m-%d %H:%M:%S")
    transformed_date = (
            datetime_obj.year * 10000000000 +
            datetime_obj.month * 100000000 +
            datetime_obj.day * 1000000 +
            datetime_obj.hour * 10000 +
            datetime_obj.minute * 100 +
            datetime_obj.second * 1
    )
    tab.append(transformed_date)

df = pd.DataFrame(data=tab)
data["date"] = df
print(data["date"])

data.to_csv("new_data.csv")


