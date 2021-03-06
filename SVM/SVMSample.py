# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:44:55 2017

@author: uyat
"""

import pandas as pd
import numpy as np
from sklearn import model_selection,preprocessing,neighbors,svm

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-9999,inplace = True)
df.drop(['id'],1,inplace=True)
X = np.array(df.drop(['class'],1))
Y = np.array(df['class'])
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size = 0.2)


clf = svm.SVC()
clf.fit(X_train,Y_train)
accuracy  = clf.score(X_test,Y_test)

print(accuracy)

#==============================================================================
# example_measure = [[5,3,2,2,1,2,1,2,4],[10,5,7,5,7,4,8,4,1]]
# 
# example_result = clf.predict(example_measure)
# 
# print(example_result)
#==============================================================================
