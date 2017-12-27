# -*- coding: utf-8 -*-
"""
Created on Fri May 19 22:21:53 2017

@author: uyat
"""

    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plotHistogram(iris_class,attr):
    plt.hist(iris_data.loc[iris_data['class'] == iris_class, attr])
    plt.show()
iris_data = pd.read_csv('iris-data.csv')
print(type(iris_data))
#cleaning data
iris_data = pd.read_csv('iris-data.csv', na_values=['NA'])

#Summary of data
#print(iris_data.describe())
#%matplotlib inline


# create a scatterplot matrix
# We have to temporarily drop the rows with 'NA' values
# because the Seaborn plotting function does not know
# what to do with them
#sb.pairplot(iris_data.dropna(), hue='class')
#Tyding data
iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
iris_data.loc[iris_data['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'

iris_data['class'].unique()
print( iris_data["class"].unique())

# This line drops any 'Iris-setosa' rows with a separal width less than 2.5 cm
iris_data = iris_data.loc[(iris_data['class'] != 'Iris-setosa') | (iris_data['sepal_width_cm'] >= 2.5)]

#==============================================================================
# plt.hist(iris_data.loc[iris_data['class'] == 'Iris-setosa', 'sepal_width_cm'])
# plt.show()
#==============================================================================

#The next data issue to address is the several near-zero sepal lengths for the Iris-versicolor rows
iris_data.loc[(iris_data['class'] == 'Iris-versicolor') &
              (iris_data['sepal_length_cm'] < 1.0)]
iris_data.loc[(iris_data['class'] == 'Iris-versicolor')]

iris_data.loc[(iris_data['class']=='Iris-versicolor') &
              (iris_data['sepal_length_cm'] <= 1.0),
              'sepal_length_cm'
              ]*=100.0
#plotHistogram('Iris-versicolor','sepal_length_cm')

#print (iris_data.describe())
#removing null alued rows
iris_data.loc[(iris_data['sepal_length_cm'].isnull() )|
        (iris_data['sepal_width_cm'].isnull()) |
         (iris_data['petal_length_cm'].isnull())|
           (iris_data['petal_width_cm'].isnull())      
        ]
#plotHistogram('Iris-setosa','petal_width_cm')

average_petal_width = iris_data.loc[(iris_data['class'] == 'Iris-setosa'),'petal_width_cm'].mean()
print (average_petal_width)
iris_data.loc[(iris_data['class'] == 'Iris-setosa') & (iris_data['petal_width_cm'].isnull())
                ,'petal_width_cm'] = average_petal_width
#plotHistogram('Iris-setosa','petal_width_cm')

#write back cleaned data to csv file
iris_data.to_csv('iris-data-clean.csv',index=False)
iris_data_clean = pd.read_csv('iris-data-clean.csv')

sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
g = sns.pairplot(iris_data_clean, hue="class")
#sns.plt.show()
#g = sns.pairplot(iris)



#Testing model
# We know that we should only have three classes
assert len(iris_data_clean['class'].unique()) == 3
# We know that sepal lengths for 'Iris-versicolor' should never be below 2.5 cm
assert iris_data_clean.loc[iris_data_clean['class'] =='Iris-versicolor','sepal_length_cm'].min() >= 2.5
# We know that our data set should have no missing measurements
assert len(iris_data_clean.loc[(iris_data_clean['sepal_length_cm'].isnull()) |
                               (iris_data_clean['sepal_width_cm'].isnull()) |
                               (iris_data_clean['petal_length_cm'].isnull()) |
                               (iris_data_clean['petal_width_cm'].isnull())]) == 0
              



#We can also make violin plots
#==============================================================================
# plt.figure(figsize=(10, 10))
# for column_index,column, in enumerate(iris_data_clean.columns):
#     if(column == 'class'):
#         continue
#     plt.subplot(2,2,column_index+1)
#     sns.violinplot(x='class',y=column,data = iris_data_clean)
# sns.plt.show()
#==============================================================================


    
        