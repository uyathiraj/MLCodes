# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:18:26 2017

@author: uyat
"""
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt
import random


def create_dataSet(size,variance,step=2,correlation=False):
    val =1
    ys=[]
    for i in range(size):
        y=val+random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype = np.float64)

#Linear regression follows y=mx+c equation
#Calculate m and b using m = (mean(x)*mean(y) - mean(x*y))/(mean(x)*mean(x)-mean(x*x))
#c = mean(y)-m*mean(x)
def best_fit_slope_and_intecept(xs,ys):
    m = ( (mean(xs)*mean(ys))-( mean(xs*ys)) ) / (mean(xs)*mean(xs)-mean(xs*xs))
    c = mean(ys)-m*mean(xs)
    return m,c

#Compute squared error coefficient using ce=1-(SE(y)/SE(mean(y)))
#SE is the squared distnace between the y and ligression line
def squared_error(ys_orign,ys_line):
    return sum((ys_line-ys_orign)**2)
    
def coefficient_of_determination(ys_origin,ys_line):
    ys_mean_line = [mean(ys_origin) for y in ys_origin]
    squared_err_reg = squared_error(ys_origin,ys_line)
    squared_err_mean = squared_error(ys_origin,ys_mean_line)
    return 1-(squared_err_reg/squared_err_mean)
    
#xs = np.array([1,2,3,4,5,6,7],dtype = np.float64)
#ys = np.array([4,6,7,8,9,12,15],dtype = np.float64)
xs,ys = create_dataSet(40,10,2,'pos')

m,c = best_fit_slope_and_intecept(xs,ys)
#print(m,c)

regression_line = [m*x+c for x in xs]

predict_x = 8

predict_y = m*predict_x+c

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color ='g')
plt.plot(xs,regression_line)
plt.show()
