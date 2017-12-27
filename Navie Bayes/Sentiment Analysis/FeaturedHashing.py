# -*- coding: utf-8 -*-
"""
Created on Wed May 24 18:14:00 2017

@author: uyat
"""

# -*- coding: utf-8 -*-


from nltk.corpus import stopwords
import re
from collections import Counter
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import FeatureHasher
from sklearn.externals import joblib
#nltk.download("stopwords") 


def preprocessText(dataset):
    #print(dataset)
    print('Preprocessing text data...........')
    stemmer = SnowballStemmer('english')
    stopwrds = set(stopwords.words('english'))
    punctuation = re.compile(r'[SUBCASE\s+(\[\])\$\\/#&%-.?!,":;()|0-9]')
    text_set_list=[]
    for document in dataset:
        document = punctuation.sub('',document)
        processed_text_list = [stemmer.stem(x) for x in document.lower().split() if x not in stopwrds]
        text_set = Counter(processed_text_list)
        text_set_list.append(text_set)
    h=FeatureHasher(n_features=1024)
    f = h.transform(text_set_list)
    return f.toarray()




def trainModel():
   print('Training model......')
   labels=[]
   text_file = open('sentiment labelled sentences/imdb_labelled.txt','r')
   lines = text_file.readlines()
   text_file.close()
   for line in lines:
       labels.append(line[len(line)-2])
   csv_file_name = 'sentiment labelled sentences/movie-pang02.csv'
   csv_df = pd.read_csv(csv_file_name)
   csv_df.loc[csv_df['class']=='Pos','class'] = 1
   csv_df.loc[csv_df['class']=='Neg','class'] = 0
   csv_df['combined'] = csv_df['text'].astype(str)+' '+csv_df['class'].astype(str)
   csv_list = csv_df['combined'].values.tolist()
   labels+=csv_df['class'].values.tolist()
   lines=lines+csv_list
   X_data = preprocessText(lines)
   Y_data = labels
   from sklearn.naive_bayes import BernoulliNB
   #from sklearn.naive_bayes import GaussianNB
   model = BernoulliNB()
   model.fit(X_data,Y_data)
   model_file = 'finalized_model.sav'
   joblib.dump(model,model_file)
   return model
   
def predictOutput(input_str):
    import os.path
    if(os.path.isfile('finalized_model.sav')):
        model = joblib.load('finalized_model.sav')
    else:
        model = trainModel()
    x_input = preprocessText(input_str)
    y_predicted = model.predict(x_input)
    return y_predicted


   
text_file = open('InputText.txt','r')
input_str = text_file.readlines()
result = predictOutput(input_str)  
print(result)    









# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
#==============================================================================
# import pandas as pd
# 
# # The entry point function can contain up to two input arguments:
# #   Param<dataframe1>: a pandas.DataFrame
# #   Param<dataframe2>: a pandas.DataFrame
# def azureml_main(dataframe1 = None, dataframe2 = None):
# 
#     # Execution logic goes here
#     print('Input pandas.DataFrame #1:\r\n\r\n{0}'.format(dataframe1))         		
#     X_data = dataframe1.drop(['label_colum', 'text_column','Preprocessed text_column'], axis=1)
#     X_data = X_data.values.tolist()
#     Y_data = dataframe1['label_colum'].values.tolist()
#     from sklearn.naive_bayes import GaussianNB
#     model = GaussianNB()
#     #model.fit(X_data,Y_data)
#     #from sklearn.model_selection import train_test_split
#     #(train_input,test_input,train_values,test_values) = train_test_split(X_data,Y_data,
#       # train_size =0.95,random_state = 1)
#     model.fit(X_data,Y_data)
#     predicted = model.predict(dataframe2)
#     #from sklearn.metrics import accuracy_score
#    # score = accuracy_score(test_values,predicted,normalize=True)
#     #res_list=[predicted,[score]]
#    # result = pd.DataFrame(res_list)
#     result= predicted
#     
#     
#     # If a zip file is connected to the third input port is connected,
#     # it is unzipped under ".\Script Bundle". This directory is added
#     # to sys.path. Therefore, if your zip file contains a Python file
#     # mymodule.py you can import it using:
#     # import mymodule
#     
#     # Return value must be of a sequence of pandas.DataFrame
#     return result
# 
#==============================================================================



















#==============================================================================
# def findUniqueWord(data):
#     unique_features =[]
#     for item in data:
#         unique_features+=item
#     return unique_features
#    
# def prepareDataSet(all_inputs,feature_set):
#     list_val = []
#     word_dict = dict.fromkeys(feature_set,0)
#    # print(word_dict)
#     for itemlist in all_inputs:
#         temp_dict = Counter(itemlist)
#         #print(temp_dict)
#         for key,value in temp_dict.items():
#             if key in word_dict.keys():
#                 word_dict[key] += value
#         list_val.append(word_dict)
#         word_dict = dict.fromkeys(word_dict,0)
#         
#     df = pd.DataFrame(list_val)
#     return df
# 
# 
# 
# def trainModelold(lines):
#    labels=[]
#    all_inputs = []       
#    for line in lines:
#        labels.append(line[len(line)-2])
#        temp_set = removeStopWords(line)
#        temp_set = stem_tokens(temp_set)
#        all_inputs.append(temp_set)
#    all_values = np.array(labels)
#    all_inputs = np.array(all_inputs)
#    unique_features = findUniqueWord(all_inputs)
#    fdist = Counter(unique_features)
#    feature_set = fdist.keys()
#    FeatureSet.feature_set = feature_set
#    df = prepareDataSet(all_inputs,feature_set)
#    #Split data
#    X_data = df.as_matrix()
#    Y_data = all_values
#    from sklearn.naive_bayes import GaussianNB
#    model = GaussianNB()
#    model.fit(X_data,Y_data)
#    return model
#==============================================================================




#==============================================================================
# def processTextData(lines):
#    labels=[]
#    all_inputs = []       
#    for line in lines:
#        labels.append(line[len(line)-2])
#        temp_set = removeStopWords(line)
#        #temp_set = stem_tokens(temp_set)
#        all_inputs.append(temp_set)
#    all_values = np.array(labels)
#    all_inputs = np.array(all_inputs)
#    unique_features = findUniqueWord(all_inputs)
#    fdist = Counter(unique_features)
#    feature_set = fdist.keys()
#    FeatureSet.feature_set = feature_set
#    
#     
# def trainModel(df):
#    X_data = df['data'].as_matrix()
#    Y_data = df['label'].as_matrix()
#    from sklearn.naive_bayes import GaussianNB
#    model = GaussianNB()
#    model.fit(X_data,Y_data)
#    return model
#==============================================================================

#==============================================================================
# def findSentiment(sentences,model):
#     pred =[]
#     test_list=[]
#     #print(FeatureSet.feature_set)
#     for line in sentences:
#         temp_set = removeStopWords(line)
#         print(temp_set)
#         test_list.append(temp_set)
#     df = prepareDataSet(test_list,FeatureSet.feature_set)
#    # print(df)
#     senti = model.predict(df.as_matrix())
#     #print(senti)
#     pred.append(senti)
#     return pred
#==============================================================================
    
    
    
    

#print( removeStopWords("There is a tree on tree",stopwrds))

#==============================================================================
# text_file = open('sentiment labelled sentences/imdb_labelled.txt','r')
# lines = text_file.readlines()
# text_file.close()
# 
# 
# labels=[]
# for line in lines:
#     labels.append(line[len(line)-2])
# all_inputs = []
# 
# for line in lines:
#    temp_set = removeStopWords(line)
#    #temp_set = stem_tokens(temp_set)
#    all_inputs.append(temp_set)
#   
# all_values = np.array(labels)
# all_inputs = np.array(all_inputs)
# 
# unique_features = findUniqueWord(all_inputs)
# fdist = Counter(unique_features)
# feature_set = fdist.keys()
# df = prepareDataSet(all_inputs,feature_set)
#==============================================================================

#==============================================================================
# 
# df.to_csv('cleaned-data.csv')
# df.dropna()
# 
# 
# #Split data
# X_data = df.as_matrix()
# Y_data = all_values
# 
# from sklearn.model_selection import train_test_split
# (train_input,test_input,train_values,test_values) = train_test_split(X_data,Y_data,
#         train_size =0.95,random_state = 1)
# 
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(train_input,train_values)
#==============================================================================




#Scoring
#==============================================================================
# from sklearn.metrics import accuracy_score
# 
# score = accuracy_score(test_values,predicted,normalize=True)
# print("Score: ",score)
# print(len([x for x in predicted if x=='1']))
#==============================================================================
#==============================================================================
# temp_pd = pd.DataFrame(data=test_input[1:,1:],index=test_input[1:,0],columns=test_input[0,1:])  
# temp_pd.to_csv('test-data.csv')
#==============================================================================
#for key,value in my_dict.items():
#    print(key,value)
#print(my_dict)
#print()

#df = pd.DataFrame.from_dict()


#==============================================================================
#==============================================================================
# import nltk
# fdist = nltk.FreqDist(unique_features)
# for word, frequency in fdist.most_common(200):
#      print(word,frequency)
#==============================================================================
#==============================================================================
#print(fdist.max)

#import nltk
#nltk.NaiveBayesClassifier.train(all_inputs)


#==============================================================================
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(all_inputs,all_values)
# test_input = ['great','awesome']
# print(model.predict(test_input))
#==============================================================================


#print(lines)
#print(csv_file)
#==============================================================================
# model = trainModelold(lines)
# 
# from sklearn.externals import joblib
# model_file = 'finalized_model.sav'
# joblib.dump(model,model_file)
# input_str = ['I think that it is a must see older John Wayne film','This movie is BAD.','Spoilers. This film is a complete mess.']
# print(findSentiment(input_str,model))
#==============================================================================

    
    

    
    
                  

                
                
            