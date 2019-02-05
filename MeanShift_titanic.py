import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing

import pandas as pd
style.use("ggplot")


df = pd.read_excel('titanic.xls' )
original_df = pd.DataFrame.copy(df)

#print(df.head())
df.drop(['body', 'name'], 1, inplace = True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace = True)
#print(df.head())
df.drop(['boat'], 1, inplace = True)
def handle_non_numeric_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = column_contents
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(convert_int, df[column]))
    return df

df = handle_non_numeric_data(df)

df.drop(['ticket', 'home.dest'], 1, inplace = True)
check = df
print(df.head())
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y =  np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_center = clf.cluster_centers_
original_df['cluster group']  = np.nan
for i in range(len(X)):
    original_df['cluster group'].iloc[i] = labels[i]

n_cluster_ = len(np.unique(labels))
print(n_cluster_)
servived_rates = {}
for i in range(n_cluster_):
    temp_df = original_df[(original_df['cluster group'] == float(i))]
    servived_cluster = temp_df[ (temp_df['survived'] == 1) ]
    servived_rate = len(servived_cluster) / len(temp_df)
    servived_rates[i] = servived_rate
print(servived_rates)
    