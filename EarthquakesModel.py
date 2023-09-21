# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:34:57 2023

@author: dell
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,median_absolute_error
import matplotlib.pyplot as plt

data=pd.read_csv('D:\\MachineCourse\\MachineLearnig\\Data\\Data\\2.4 SVR\\Earthquakes.csv')
print(data.head())
#=================== clean data =========================#
data=data.dropna()
sc=StandardScaler()
#========================= analysis data =======================#
print(data.info)
print(data.describe())
print(data.isnull().sum())
print(data.value_counts())

x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print(x.head())
print(y.head())
#======================== split data ====================#
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
#x_train=sc.fit_transform(x_train)
#x_test=sc.fit_transform(x_test)
model=SVR(kernel='linear',)
model.fit(x,y)
y_predict=model.predict(x_test)

print(y_predict)
print(f"mean absolute error  = {mean_absolute_error(y_test,y_predict)}") # 11.858958486124413
print(f"median absolute error  = {median_absolute_error(y_test,y_predict)}") # 5.800059761996323





plt.scatter(y_test,y_predict)
plt.show()












