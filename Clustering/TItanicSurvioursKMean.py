# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:08:37 2017

@author: uyat
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
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
    #print(df.head())
    return df

df = pd.DataFrame()
df = pd.read_excel('titanic.xls')
df.fillna(0,inplace = True)
#print(df.head())
df.drop(['name','body'],1,inplace = True)
df.apply(pd.to_numeric, errors='ignore')
#df.convert_objects(convert_numeric = True)
#print(df.head())
#g = df.columns.to_series().groupby(df.dtypes).groups
#print(g)
df = handle_non_numericalValue(df)
#df.drop('sibsp',1,inplace = True)
print(df.head())

clf = KMeans(n_clusters= 2)
X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
Y = np.array(df['survived'])

clf.fit(X)
counter =0
predictions =[] 
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    #print(prediction)
    predictions.append(prediction)
    if prediction == Y[i]:
        counter+=1

print("Accurancy ",counter/len(X))
#print(prediction[0])



