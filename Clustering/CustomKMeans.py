# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 20:45:02 2017

@author: uyat
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn import preprocessing

class K_Mean:
    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter = max_iter
        
    def fit(self,data):
        self.classifications={}
        self.centroids={}
        
        #Make first k points as centroids
        for i in range(self.k):
            self.centroids[i]=data[i]
        
        
        for i in range(self.max_iter):
            self.classifications={}
            #Iniitialize classifier
            for g in range(self.k):
                self.classifications[g] =[]
            
            #Find the distance of all point to the centroids
            for feature in data:
                distances =[[np.linalg.norm(feature-self.centroids[centroid])] for centroid in self.centroids]
                classification = distances.index(min(distances))
                #print(classification)
                #print(feature)
                self.classifications[classification].append(feature)
            
            #Self.classification has set of all the points closer to the centoid i
            #print(self.classifications)
            #Save copy of old centroid
            prev = dict(self.centroids)
            
            #find the new centroid point among those points which is the average of all points
            for classification in self.classifications:
                self.centroids[i] = np.average(self.classifications[classification],axis=0)
            
            optimized = True
            
            for i in  self.centroids:
                if np.sum((self.centroids[i] - prev[i])/(prev[i]*100)) >= self.tol:
                    optimized = False
                    break
                
            if(optimized):
                break
        
        colors =['r','b','k']
        for classification in self.classifications:
            color = colors[classification]
            for feature in self.classifications[classification]:
                #print(feature)
                plt.scatter(feature[0],feature[1],c = color,s=150)

        for centroid in self.centroids:
            col = colors[centroid]
            plt.scatter(self.centroids[centroid][0],self.centroids[centroid][1],s = 150,c = col,marker ="x",linewidths=5)
        #plt.show()
    
    
    
    def predict(self,data):
        distances = [[np.linalg.norm(data-self.centroids[centroid])] for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
    



def TitanicSurvivor():
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
    X = np.array(df.drop(['survived'],1).astype(float))
    X = preprocessing.scale(X)
    Y = np.array(df['survived'])
    print(df.head())
    clf = K_Mean()
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
    
    
def initial_test():
    data = np.array([[1,2],
                 [1.5,1.8],
                 [5,8],
                [8,8],
                 [1,0.6],
                 [9,11]])
    unknowns = np.array([[1,5],
                     [4,6],
                     [3,1],
                     [2,-6],
                     [9,11],
                     [5,8]])
    clf = K_Mean()
    clf.fit(data)
    print(clf.classifications)
    colors =['r','b','k']
    for unknown in unknowns:
        predicted = clf.predict(unknown)
        plt.scatter(unknown[0],unknown[1],c = colors[predicted],s = 150,marker ="*")
        print(predicted)
    plt.show()





############################  
initial_test()
#TitanicSurvivor()


