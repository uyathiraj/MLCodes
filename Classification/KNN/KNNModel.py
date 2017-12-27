# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 19:21:40 2017

@author: uyat
"""
import numpy as np
import warnings
from collections import Counter
import pandas as pd

#[[plt.scatter(ii[0],ii[1],s=100,c = i) for ii in data_set[i]] for i in data_set]
#plt.show()

def k_nearest_neighbors(data,predict,k=3):
    if(k<=len(data)):
        warnings.warn('K value is set to less than that of data size')
        return
    distances=[]
    for group in data:
        for features in data[group]:
            euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidian_distance,group])
    #print(distances)
    distances = sorted(distances)
    #print(distances)
    voted_list = [v[1] for v in distances[:k]]
    #print(voted_list)
    vote_result = Counter(voted_list).most_common(1)
    #print(vote_result[0][0])
    confidence = vote_result[0][1]/k
    return vote_result[0][0],confidence
    

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-9999,inplace=True)
df.drop(['id'],1,inplace = True)
df = df.astype(float)
categories = set(df['class'])
print(categories)
#Shuffle data frame data
df = df.sample(frac=1)


test_size =0.2
test_index = int(len(df)*test_size)
print('Len',len(df))
print('Test index',test_index)              
test_data = {}
test_x_data=[]
test_data ={}
train_data={}
for i in categories:
    temp = df.loc[df['class'] == i]
    temp = temp.drop(['class'],1)
    temp_index = len(temp)- test_index
    train_data[i]=temp[:temp_index].values.tolist()
    test_data[i] = temp[:temp_index].values.tolist()
    #test_x_data += temp[temp_index:].values.tolist()

#print(train_data)
#print(test_x_data)
#print(len(train_data[2.0]))
result=[]
confidence =[]
res_actual =[]
for group in test_data:
    for feature in test_data[group]:
        res_actual.append(group)
        res,conf = k_nearest_neighbors(train_data,feature,25)
        confidence.append(conf)
        result.append(res)
        
print(Counter(result))        

#==============================================================================
# for predict in test_data:
#     
#     res = k_nearest_neighbors(train_data,predict,3)
#     result.append(res)
#==============================================================================


#print(res_actual)
#print('#'*20)
#print(result)

    
total = len(res_actual)
correct = 0

for i in range(len(res_actual)):
    if result[i] == res_actual[i]:
        correct+=1
    else:
        print(confidence[i])

accurancy = float(correct)/float(total)
print(accurancy)






    


     
    


