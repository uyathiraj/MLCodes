# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:47:20 2017

@author: uyat
"""

import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

centeres = [[1,1,1],[5,5,5],[3,10,10]]
X,_ = make_blobs(n_samples = 100, centers = centeres,cluster_std = 1)

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centeres = ms.cluster_centers_
print(cluster_centeres)
print(labels)
no_clusters = len(np.unique(labels))
print(no_clusters)


colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], s =150, c = colors[labels[i]] ,marker = "*")

ax.scatter(cluster_centeres[:,0],cluster_centeres[:,1],cluster_centeres[:,2], s = 150, c = "k",marker="x")

plt.show()