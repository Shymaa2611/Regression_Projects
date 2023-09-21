# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 23:23:25 2023

@author: shymaa
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import seaborn as sns
import numpy as np
data=pd.read_csv('D:\\MachineCourse\\MachineLearnig\\Data\\Data\\2.1 Linear Regression\\houses.csv')

print(data.head())
data=data.dropna()
print(data.shape)
print(data.isnull())
print(data.value_counts())
print(data.info())
print(data.describe())

x=data.iloc[:,:-1]
#============================= x values ===================#
print(x)
y=data['price']
#============================= y values ===================#
print(y)

#======================= train and test model =====================#
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

linearRegression=LinearRegression()
#================ predict theta value ==========#
linearRegression.fit(x_train,y_train)
#============= predict model =============#
y_predict=linearRegression.predict(x_test)
y_predict=pd.Series(y_predict)
print(type(y_predict))



print(f"accuracy train = {linearRegression.score(x_train, y_train)}")
print(f"accuracy test = {linearRegression.score(x_test, y_test)}")
print(metrics.mean_absolute_error(y_test,y_predict))
print(metrics.mean_squared_error(y_test,y_predict))
print(metrics.median_absolute_error(y_test,y_predict))

#print(f"accuracy train = {linearRegression.score(y_test, y_predict)}")
#cm=metrics.confusion_matrix(y_test, y_predict)
#sns.heatmap(cm,center=True)
plt.scatter(y_test,y_predict)
plt.plot(y_test,y_predict,color='red')
plt.show()





















