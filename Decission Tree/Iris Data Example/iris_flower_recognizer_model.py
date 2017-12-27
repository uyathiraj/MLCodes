# -*- coding: utf-8 -*-
"""
Created on Sun May 21 17:43:30 2017

@author: uyat
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris_data_clean = pd.read_csv('iris-data-clean.csv')


# We're using all four measurements as inputs
# Note that scikit-learn expects each entry to be a list of values, e.g.,
# [ [val1, val2, val3],
#   [val1, val2, val3],
#   ... ]
# such that our input data set is represented as a list of lists
#print (iris_data_clean.describe())
# We can extract the data in this format from pandas like this:
all_inputs = iris_data_clean[['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm']].values
all_values = iris_data_clean['class'].values
#print (all_inputs)

# Make sure that you don't mix up the order of the entries
# all_inputs[5] inputs should correspond to the class in all_classes[5]

# Here's what a subset of our inputs looks like:
#print (all_inputs[:5])

from sklearn.model_selection import train_test_split
(train_input,test_input,train_values,test_values) = train_test_split(all_inputs,all_values,
        train_size =0.75,random_state = 1)

from sklearn.tree import DecisionTreeClassifier
#Create decision tree
decision_tree_classifier = DecisionTreeClassifier()

#Train the classifier on the training set
decision_tree_classifier.fit(train_input,train_values)

#Validate the classifier on testing data 
print (decision_tree_classifier.score(test_input,test_values))

#==============================================================================
# #==============================================================================
from sklearn.externals.six import StringIO  
from sklearn import tree
import pydotplus
dot_data = StringIO()
tree.export_graphviz(decision_tree_classifier,out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf('iris-tree.pdf')
# #==============================================================================
#==============================================================================

#==============================================================================
#==============================================================================
#==============================================================================
# model_accuracies = []
# for repetition in range(1000):
#     (train_input,test_input,train_values,test_values) = train_test_split(all_inputs,all_values,
#          train_size =0.75)
#     decision_tree_classifier = DecisionTreeClassifier()
#     decision_tree_classifier.fit(train_input,train_values)
#     model_accuracies.append(decision_tree_classifier.score(test_input,test_values))
#==============================================================================
#==============================================================================
    
#print(model_accuracies)
#==============================================================================
# sns.distplot(model_accuracies,kde = True)
# sns.plt.show()
#==============================================================================
#==============================================================================
#k-fold cross-validation
import numpy as np
from sklearn.model_selection import StratifiedKFold
#==============================================================================
# def plot_cv(cv):
#     masks=[]
#     for train,test in cv:
#         mask = np.zeros(n_samples,dtype = bool)
#         mask[test] = 1
#         masks.append(mask)
#     plt.figure(figsize=(15,15))
#     plt.imshow(masks,interpolation='none')
#     plt.ylabel('Fold')
#     plt.xlabel('Row #')
#==============================================================================
    
#plot_cv(StratifiedKFold(all_values,n_folds=10),len(all_values))
#==============================================================================
# skf= StratifiedKFold(n_splits = 10)
# #plot_cv(skf,)
# print(skf.get_n_splits(all_inputs,all_values))
# print(skf)
# model_accuracies = []
# for train,test in skf.split(all_inputs,all_values):
#    # print("TRAIN: ",len(train)," TEST: ",len(test))
#     train_input,train_values,test_input,test_values = all_inputs[train],all_values[train],all_inputs[test],all_values[test]
#     #print(train_input,train_values,test_input,test_values)
#     decision_tree_classifier = DecisionTreeClassifier()
#     decision_tree_classifier.fit(train_input,train_values)
#     model_accuracies.append(decision_tree_classifier.score(test_input,test_values))
# print(model_accuracies)
# sns.distplot(model_accuracies,kde = True)
# sns.plt.title('average score'+str(np.mean(model_accuracies)))
# sns.plt.show()
#==============================================================================




#Parameter tuning(tuning decision tree classfier)
from sklearn.model_selection import GridSearchCV
parameter_grid = {'max_depth':[1,2,3,4,5],
            'max_features':[1,2,3,4]}
#==============================================================================
# parameter_grid = {'criterion': ['gini', 'entropy'],
#                   'splitter': ['best', 'random'],
#                   'max_depth': [1, 2, 3, 4, 5],
#                   'max_features': [1, 2, 3, 4]}
#==============================================================================
skf= StratifiedKFold(n_splits = 10)
decision_tree_classifier = DecisionTreeClassifier()
cross_validation = skf.get_n_splits(all_inputs,all_values)
grid_search = GridSearchCV(decision_tree_classifier,
                           param_grid = parameter_grid,
                           cv = cross_validation)
grid_search.fit(all_inputs,all_values)
print('Best parameter ',grid_search.best_score_)
print('Best parameter ',grid_search.best_params_)

#Visualizing the parameter tuning
grid_visualization =[]
for grid_pair in grid_search.grid_scores_:
    grid_visualization.append(grid_pair.mean_validation_score)
    
grid_visualization = np.array(grid_visualization)
grid_visualization.shape = (5,4)
sns.heatmap(grid_visualization,cmap='Blues')
#plt.xticks()
plt.xticks(np.arange(4) + 0.5, grid_search.param_grid['max_features'])
plt.yticks(np.arange(5) + 0.5, grid_search.param_grid['max_depth'][::-1])
plt.xlabel('max_features')
plt.ylabel('max_depth')
sns.plt.show()
