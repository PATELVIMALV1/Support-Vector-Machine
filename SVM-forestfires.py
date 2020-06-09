# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:44:47 2020

@author: patel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\patel\\Downloads\\forestfires.csv")

data.head()

data.dtypes



#understanding the problem
#since here we need to predict the area burt during the forest fire ,so the day,month are 
#irrelevant here .

data2=pd.read_excel("C:\\Users\\patel\\OneDrive\\forestfires.xlsx")

#a pair plot 
sns.pairplot(data=data2)

from sklearn.preprocessing import StandardScaler
norm=StandardScaler()  
norm.fit(data2)
norm_data=norm.transform(data2) 
 
col_names=data.columns

y=data2.iloc[:,9]

x=data2.iloc[:,0:9] 

'''
plt.(x=, height, kwargs)
'''

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)
train_x.shape
train_y.shape
from sklearn import svm


model_linear= svm.SVC(kernel='linear', C=1 ,gamma='auto').fit(train_x,train_y)

pred_test_linear = model_linear.predict(test_x)

np.mean(pred_test_linear==test_y)


#polynomial model
r=[2,3,4,5,6]
for i in  range(len(r)):
    model_poly=svm.SVC(kernel='poly',C=1,gamma='auto',degree=i).fit(train_x,train_y)
    pred_test_poly = model_poly.predict(test_x)
    print(model_poly)
    print(np.mean(pred_test_poly==test_y))
    
#rbf model
r=[1,10,100,1000]
for i in  range(len(r)):
    gmodel_rbf = svm.SVC(kernel = "rbf",gamma=r[i],C=0.5)
    gmodel_rbf.fit(train_x,train_y)
    pred_test_rbf = gmodel_rbf.predict(test_x)
    print(gmodel_rbf)
    print(np.mean(pred_test_rbf==test_y) )
    
#incresing of gamma value leads to low accuracy model

c=[1,10,100,1000,10000]
for i in  range(len(c)):
    model_rbf =svm.SVC(kernel = "rbf",gamma=10,C=c[i])
    model_rbf.fit(train_x,train_y)
    pred_test_rbf = model_rbf.predict(test_x)
    print(model_rbf)
    print(np.mean(pred_test_rbf==test_y) )
    
    
    
    




