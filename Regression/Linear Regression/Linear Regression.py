# -*- coding: utf-8 -
"""
Created on Wed May 31 11:40:58 2017

@author: uyat
"""

import pandas as pd
import math
import numpy as np
from sklearn import preprocessing,cross_validation
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pickle
#import quandl

#df = quandl.get('WIKI/GOOGL')
#df.to_csv('quandleData1.csv')
df = pd.read_csv('quandleData1.csv')
df.dropna(inplace = True)
#print(df.head())

df = df[['Date','Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = ((df['Adj. High']-df['Adj. Close'])/df['Adj. Close'])*100.00
df['PCT_Change'] = ((df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'])*100.00

df_date_Index = df['Date']
df_date_Index.dropna(inplace = True)
df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]
forecast_col = 'Adj. Close'

df.fillna(-9999,inplace=True)

#Compute the frequency to be shifted
forecast_out = int(math.ceil(0.01*len(df)))

#Create label column to predict forecast col value of next forecast_out days
df['label'] = df['Adj. Close'].shift(-forecast_out)

#drop rows of shifted values
X = np.array(df.drop('label',1))

X = preprocessing.scale(X)

X_lately = X[-forecast_out:]

X = X[:-forecast_out]

df_dropped = df.loc[df['label'].isnull()]

df.dropna(inplace =True)

Y = np.array(df['label'])

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)

#Uncomment folowing block while running first time
#==============================================================================
clf = LinearRegression(n_jobs=1)
clf.fit(X_train,Y_train)
 #Save trained model into local file using pickle
with open('linearRegressionModel.pickle','wb') as f:
    pickle.dump(clf,f)
#==============================================================================

pickle_in = open('linearRegressionModel.pickle','rb')
clf = pickle.load(pickle_in)

accurancy = clf.score(X_test,Y_test)
print(accurancy)

forecast_set = clf.predict(X_lately)
#print(forecast_set)   
df = df.append(df_dropped)
df['Forecast'] = np.nan
df.iloc[-forecast_out:,df.columns.get_loc('Forecast')] = forecast_set
        
df['Date'] = df_date_Index
df = df.set_index('Date')
#print(df.tail(40))  
df.to_csv('ForecastResult1.csv')
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show() 










