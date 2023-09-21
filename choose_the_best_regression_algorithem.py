# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:23:45 2023

@author: SHYMAA
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,SGDRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
path=input("Enter path ? ")
data=pd.read_csv(path)
#================= data clean ============================#
data=data.dropna()
x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print("===================== data =================")
print(data.head())
print("===================== X =================")
print(x.head())
print("===================== Y =================")
print(y.head())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
#=========================== scalling model ============================#
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.fit_transform(x_test)
#======================== train model ===========================#

algorithms=[
    
    ("LinearRegression",LinearRegression()),
    ("Ridge",Ridge()),
    ("Lasso",Lasso()),
    ("SGD",SGDRegressor()),
    ]

error_score=[]


for name,algorithm in algorithms:
    model=algorithm
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    error_value=metrics.mean_absolute_error(y_test, y_predict)
    error_score.append((name,error_value))

#print(error_score[0][0])
print(error_score)
min_value=error_score[0][1]
best_model=' '
for name,error in error_score:
    if min_value>=error:
        min_value=error
        best_model=name
        
        
print(f"[ best algorithm for this data is : {best_model} , error  is {min_value}")

print("=============================== Prediction =======================")

if best_model=="LinearRegression":
        model=LinearRegression(normalize=True)
        model.fit(x_train,y_train)
        y_predict=model.predict(x_test)
        print(y_predict)
elif best_model=="Ridge":
    model=Ridge()
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    print(y_predict)
        
elif best_model=="Lasso":
    model=Lasso()
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    print(y_predict)
        
else:
    model=SGDRegressor()
    model.fit(x_train,y_train)
    y_predict=model.predict(x_test)
    print(y_predict)
    
        






















