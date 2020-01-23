#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:38:13 2019

@author: priyanshutuli
"""

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

dataset = pd.read_csv('Downloads/boston.csv')
print(dataset.head(8))
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
print(X)
print(dataset.shape)
print(y)

print(X.isnull().sum())
print(y.isnull().sum())

#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(y,bins=30)
plt.show()

correlation_matrix = dataset.corr().round(2)
print(correlation_matrix)
sns.heatmap(data=correlation_matrix, annot=True)
plt.figure(figsize=(20, 5))

Q1_x= X.quantile(0.25)
Q3_x= X.quantile(0.75)
mean1=X.mean()
mean2=y.mean()
print(Q1_x,Q3_x)
IQR_x = Q3_x - Q1_x
print(IQR_x)
#print(dataset < (Q1 - 1.5 * IQR) |(dataset > (Q3 + 1.5 * IQR)))
print(X < (Q1_x - 1.5 * IQR_x),(X > (Q3_x + 1.5 * IQR_x)))
X_out = X[~((X < (Q1_x - 1.5 * IQR_x)) |(X > (Q3_x + 1.5 * IQR_x))).any(axis=1)]
X_out.shape

Q1_y =y.quantile(0.25)
Q3_y=y.quantile(0.75)
IQR_y=Q3_y-Q1_y
y_out=y[~((y < (Q1_y - 1.5 * IQR_y)) |(y > (Q3_y + 1.5 * IQR_y))).any(axis=2)]
y_out.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#fig, ax = plt.subplots(figsize=(16,8))
#ax.scatter(dataset['lstat'],dataset['rm'])
#plt.show()

#from scipy import stats
#z = np.abs(stats.zscore(dataset))
#print(z)
#threshhold=3
#print(np.where(z>3))



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
print(regressor.r2_score)
#X=np.append(arr=np.ones((506,1)),values=X,axis=1)
import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(y_train,X_train).fit()
print(regressor_OLS.summary())
y_train_pred=regressor_OLS.predict(X_train)
print(regressor_OLS.rsquared,regressor_OLS.rsquared_adj)

from sklearn.metrics import mean_squared_error
rmse_train=(np.sqrt(mean_squared_error(y_train,y_train_pred)))
print(rmse_train)

from sklearn.metrics import r2_score
r2_train = r2_score(y_train, y_train_pred)
print(r2_train)

#regressor_OLS = sm.OLS(y_test,X_test).fit()
#print(regressor_OLS.summary())
y_test_pred=regressor_OLS.predict(X_test)
print(regressor_OLS.rsquared,regressor_OLS.rsquared_adj)
rmse_test=(np.sqrt(mean_squared_error(y_test,y_test_pred)))
print(rmse_test)
r2_test=r2_score(y_test,y_test_pred)
print(r2_test)
#print(rmse_test,r2_test)


#from sklearn.metrics import mean_squared_error
#rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
#print(rmse,r2)

#y_test_pred = regressor.predict(X_test)
#rmse1 = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
#r2_1 = r2_score(y_test, y_test_pred)
#print(rmse1,r2_1)
