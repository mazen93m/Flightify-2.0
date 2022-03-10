#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:39:25 2022

@author: ajgray
"""

import cleaning_tool as CT
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import math

# reading in the data csv file outputed by the cleaning tool
data = CT.data_dict['FAI']

sns.set()
correlation_matrix = data[['VFR','AWND','SNOW','SNWD','TMAX','isAHoliday','PRCP']].corr()

sns.heatmap(correlation_matrix)
plt.show()

sns.scatterplot(x='VFR',y='PRCP',data=data)
plt.xlabel('Daily VFR Traffic')
plt.ylabel('Daily Precipitation')
plt.title('Daily Precipitation by VFR Traffic')
plt.show()

sns.scatterplot(x='VFR',y='TMAX',data=data)
plt.xlabel('Daily VFR Traffic')
plt.ylabel('Daily Maximum Temperature')
plt.title('Maximum Temp by VFR Traffic')
plt.show()

data_np = data.to_numpy()

# dataset to train and test 
X = data_np[:, -8:-1]

# dependent label variable
y = data_np[:, -9]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Instantiate a LinearRegression model with default parameter values
linReg = LinearRegression()
 
# Fit linReg to the train set
linReg.fit(X_train, y_train)
 
y_pred = linReg.predict(X_test)

# Get model performance
print('Coefficients: ',[round(i,2) for i in linReg.coef_])
print('Intercept:',round(linReg.intercept_,2))
print('Root Mean squared error (MSE): %.2f'% math.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('Coefficient of determination (R^2): %.2f'% r2_score(y_test,y_pred))

sns.scatterplot(x=y_test,y=y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()
