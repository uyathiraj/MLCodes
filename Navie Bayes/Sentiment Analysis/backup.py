# -*- coding: utf-8 -*-
"""
Created on Tue May 23 20:07:01 2017

@author: uyat
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:12:07 2017

@author: uyat
"""

from nltk.corpus import stopwords
import numpy as np
import re
from collections import Counter
import pandas as pd
from nltk.stem.porter import PorterStemmer
#nltk.download("stopwords") 

class MyCalss:
    
#==============================================================================
# def preprocess(sentence):
#     sentence = sentence.lower()
#     tokenizer = RegexpTokenizer(r'\w+')
#     tokens = tokenizer.tokenize(sentence)
#     filtered_words = [w for w in tokens if not w in stopwords.words('english')]
#     return filtered_words
#==============================================================================
    feature_set ={}    
    def stem_tokens(tokens):
        stemmer = PorterStemmer()
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed
    
    #Remove stop words from sentence
    def removeStopWords(sentence):
    #print(sentence)
        stopwrds = set(stopwords.words('english'))
        punctuation = re.compile(r'[(\[\])\$\\/#&%-.?!,":;()|0-9]')
        sentence = punctuation.sub('',sentence)
        #sentence = ''.join(e for e in sentence if e.isalnum()| e.isspace())  
        processed_text_set ={}
        processed_text_set = {x for x in sentence.lower().split() if x not in stopwrds}
        
        return processed_text_set
    
    
    def findUniqueWord(data):
        unique_features =[]
        for item in data:
            unique_features+=item
        return unique_features
       
    def prepareDataSet(all_inputs,feature_set):
        list_val = []
        word_dict = dict.fromkeys(feature_set,0)
       # print(word_dict)
        for itemlist in all_inputs:
           temp_dict = Counter(itemlist)
           #print(temp_dict)
           for key,value in temp_dict.items():
               word_dict[key] += value
        list_val.append(word_dict)
        word_dict = dict.fromkeys(word_dict,0)
        df = pd.DataFrame(list_val)
        return df
    
    
    
    def trainModel(lines):
       labels=[]
       all_inputs = []       
       for line in lines:
           labels.append(line[len(line)-2])
           temp_set = MyCalss.removeStopWords(line)
           #temp_set = stem_tokens(temp_set)
           all_inputs.append(temp_set)
       all_values = np.array(labels)
       all_inputs = np.array(all_inputs)
       unique_features = MyCalss.findUniqueWord(all_inputs)
       fdist = Counter(unique_features)
       MyCalss.feature_set = fdist.keys()
       df = MyCalss.prepareDataSet(all_inputs,MyCalss.feature_set)
       #Split data
       X_data = df.as_matrix()
       Y_data = all_values
       from sklearn.model_selection import train_test_split
       (train_input,test_input,train_values,test_values) = train_test_split(X_data,Y_data,
             train_size =0.75,random_state = 1)
       from sklearn.naive_bayes import GaussianNB
       model = GaussianNB()
       model.fit(train_input,train_values)
       return model
       
    
    def findSentiment(sentences,model):
        senti = []
        print(MyCalss.feature_set)
        for line in lines:
            temp_set = MyCalss.removeStopWords(line)
            df = MyCalss.prepareDataSet(temp_set,MyCalss.feature_set)
            tempSent = model.predict(df.as_matrix())
            senti.append(tempSent)
        return senti
    
    
    

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

text_file = open('sentiment labelled sentences/imdb_labelled.txt','r')
lines = text_file.readlines()
text_file.close()
model = MyCalss.trainModel(lines)

input_str = ["Good movie"]
print(MyCalss.findSentiment(input_str,model))

    
    

    
    
                  

                
                
            