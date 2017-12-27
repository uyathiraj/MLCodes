# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 19:59:20 2017

@author: uyat
"""

import matplotlib.pyplot as plt
import numpy as np





class SVM_Classifier:
        
    def __init__(self,visualization=True):
        print('insidde init')
        self.visualization = visualization
        self.colors={1:'r',-1:'g'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
            
    def fit(self,data):
        self.data = data
        opt_dict={}
        #||w||=[wt,b]
        all_data=[]
        transformation =[[1,1],[-1,1],[-1,-1],[1,-1]]
        for yi in self.data:
            for d in self.data[yi]:
                for feature in d:
                    all_data.append(feature)
        print(all_data)    
       # print(all_data)
        self.max_feature_val = max(all_data)
        self.min_feature_val = min(all_data)
        all_data = None
        print(self.max_feature_val,"  ",self.min_feature_val)
        step_size = [self.max_feature_val*0.1,
                     self.max_feature_val*0.01,
                     self.max_feature_val*0.001]
        b_range_multiple = 10
        
        b_multiple=2
        
        latest_optimum = self.max_feature_val*10
        
        
        for step in step_size:
            
            w= np.array([latest_optimum,latest_optimum])
            optimized = False
            
            while not optimized:
                for b in np.arange(-1*(self.max_feature_val*b_range_multiple),
                                   self.max_feature_val*b_range_multiple,b_multiple):
                    for transform in transformation:
                        w_t = w*transform
                        found_option = True
                        #print(w_t,b)
                        for yi in self.data:
                            for xi in self.data[yi]:
                                #print(w_t,xi,yi,b)
                                res = yi*(np.dot(w_t,xi) + b)
                               # print(res)
                                if not res>=1:
                                    found_option = False
                                    break
                            if not found_option:
                               break
                        if found_option :
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] <0:
                    optimized = True
                    print("Optimized a step")
                else:
                    w=w-step
                #print (opt_dict)
            
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w  = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum =  opt_choice[0][0]+step*2

    def predict(self,data):
        #sign(wx+b)
        classfication = np.sign(np.dot(np.array(data),self.w)+self.b)
        if self.visualization and classfication !=0:
            self.ax.scatter(data[0],data[1],s=200,marker="*", c=self.colors[classfication])
            
        return classfication
        
    
    def visualize(self,data):
        [[self.ax.scatter(x[0],x[1],s=100,color = self.colors[i]) for x in data[i]] for i in data] 
        
        def hyperPlane(x,w,b,v):
            return(-w[0]*x-b+v)/w[1]
        datarange = (self.min_feature_val*.9,self.max_feature_val*1.1)
        hype_x_min = datarange[0]
        hype_x_max = datarange[1]
        
        #(w.x+b) = 1
        psv1 = hyperPlane(hype_x_min,self.w,self.b,1)
        psv2 = hyperPlane(hype_x_max,self.w,self.b,1)
        self.ax.plot([hype_x_min,hype_x_max],[psv1,psv2],'k')
        
        #(w.x+b) = -1
        nsv1 = hyperPlane(hype_x_min,self.w,self.b,-1)
        nsv2 = hyperPlane(hype_x_max,self.w,self.b,-1)
        self.ax.plot([hype_x_min,hype_x_max],[nsv1,nsv2],'k')
        
        #(w.x+b) = 0
        db1 = hyperPlane(hype_x_min,self.w,self.b,0)
        db2 = hyperPlane(hype_x_max,self.w,self.b,0)
        self.ax.plot([hype_x_min,hype_x_max],[db1,db2],'y--')
        
        plt.show()
        

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}
#data_set={1:np.array([[1,2],[3,6],[2,4],[3,8]]),
#          -1:np.array([[1,5],[2,5],[3,2],[8,3]])}
clf = SVM_Classifier()
clf.fit(data_dict)
predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8],
              [1,7],
              [5,1]]
for p in predict_us:
    print(clf.predict(p))
clf.visualize(data_dict)