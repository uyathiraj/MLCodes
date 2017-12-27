# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:32:46 2017

@author: uyat
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MeanShift
from sklearn import preprocessing


def handle_non_numericalValue(df):
    #Convert string data into numeric using Encoder
    enc = LabelEncoder()
    columns_list = df.columns.values
    for col in columns_list:
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            unique_item = df[col].tolist()
            enc.fit(unique_item)
            df[col] = enc.transform(unique_item)
    return df

df = pd.DataFrame()
df = pd.read_excel('titanic.xls')

old_df = df.copy(deep=True)

df.fillna(0,inplace = True)
df.drop(['name','body'],1,inplace = True)
df.apply(pd.to_numeric, errors='ignore')

df = handle_non_numericalValue(df)
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)
labels = clf.labels_
n_cluster = len(np.unique(labels))
print(n_cluster)
old_df['cluster_group'] = labels
#print(old_df.head())  

survive_rate={}

for i in range(n_cluster):
    cluster_count = len(old_df[old_df['cluster_group']==i])
    survive_rate[i] = cluster_count/len(old_df)

print(survive_rate)

#print( old_df[old_df['cluster_group'] == 3].describe() )

cluster_0 = old_df[old_df['cluster_group'] == 3]
#print(cluster_0.head())
cluster_0_fc = cluster_0[cluster_0['pclass']== 1]

print(cluster_0_fc.describe())



    

    
    




