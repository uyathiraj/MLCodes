# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 17:55:46 2017

@author: uyat
"""

import numpy as np
import matplotlib.pyplot as plt

class MeanShift:
    def __init__(self,radius=4):
        self.radius = radius
    
    def fit(self,data):
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]
            
        while True:
            new_centroids=[]
            for i in centroids:
                in_bandwidth=[]
                centroid = centroids[i]
                for feature in data:
                    d = np.linalg.norm(feature-centroid)
                    if d <= self.radius:
                        in_bandwidth.append(feature)
                        
                new_centroid =  tuple(np.average(in_bandwidth,axis=0))
                new_centroids.append(new_centroid)
                #print(new_centroids)
                
            uniques = sorted(list(set(new_centroids)))
                
            prev_centroids = dict(centroids)
            centroids={}    
            for i in range(len(uniques)):
                centroids[i] = uniques[i]
            
            if np.array_equal(prev_centroids,centroids):
                break
        
        self.centroids = centroids

    def predict(self,data):
        centroids = self.centroids
        
        distances = [np.linalg.norm(centroids[c]-data) for c in centroids]
        
        for d in sorted(distances):
            if d <= self.radius:
                classification = distances.index(d)
                
        
        
        #classification = distances.index(min(distances))
        return classification
            
                
 





data = np.array([[1,2],
                [1.5,1.8],
                [5,8],
                [8,8],
                [1,0.6],
                [9,11],
                [-10,-5]])
clf = MeanShift()
clf.fit(data)
plt.scatter(data[:,0],data[:,1],s = 150,c = "r")

centroids = clf.centroids
for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],s = 150, c = 'k', marker = "*")
    #print(c)

for d in data:
    print(clf.predict(d))
plt.show()



         
            
                        
                
            
                    
        
        