#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:39:25 2022

@author: ajgray
"""

import dataCleaner as d
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import math

import pickle


#code adjusted by Kim
#start model_dict

model_dict = {}

for key in d.datasets:
        
    #code adjusted by Kim
    # reading in the data csv file outputed by the cleaning tool
    #data = d.datasets['ANC']
    data = d.datasets[key]
    print(data['LOC'][0])
 
    
    sns.set()
    correlation_matrix = data[['VFR','AWND','SNOW','TMAX','isAHoliday','PRCP']].corr()
    
    sns.heatmap(correlation_matrix)
    plt.show()
    
    #Kim Added code for histogram for VFR
    data['VFR'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
    axis=plt.title('VFR Counts for ' + key)
    plt.xlabel('Counts')
    plt.ylabel('VFR Counts')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    #Kim switched axis labels to make target on y axis
    sns.scatterplot(x='PRCP',y='VFR',data=data)
    plt.ylabel('Daily VFR Traffic')
    plt.xlabel('Daily Precipitation')
    plt.title('Daily VFR Traffic vs Precipitation')
    plt.show()
    
    sns.scatterplot(x='TMAX',y='VFR',data=data)
    plt.ylabel('Daily VFR Traffic')
    plt.xlabel('Daily Maximum Temperature')
    plt.title('VFR Traffic vs Maximum Temperature')
    plt.show()
    
    data_np = data.to_numpy()
    
    # dataset to train and test 
    #X = data_np[:, -8:-1]
    
    #code added by Kim
    X = data_np[:, [-8,-7,-5,-4,-2,-1]]
    
    # dependent label variable
    y = data_np[:, -9]
    
    
    #code added by Kim
    #X = pd.DataFrame(data, columns=['IFR', 'AWND', 'PRCP_SQRT', 'SNOW', 'TMAX', 'isAHoliday'])
    #y = pd.DataFrame(data, columns=['VFR'])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    
    # Instantiate a LinearRegression model with default parameter values
    
    linReg = LinearRegression()
     
    # Fit linReg to the train set
    linReg.fit(X_train, y_train)
     
    y_pred = linReg.predict(X_test)
    
    # Get model performance
    coeff = [round(i,2) for i in linReg.coef_]
    print('Coefficients: ',[round(i,2) for i in linReg.coef_])
    interc = round(linReg.intercept_,2)
    print('Intercept:',round(linReg.intercept_,2))
    print('Root Mean squared error (MSE): %.2f'% math.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    print('Coefficient of determination (R^2): %.2f'% r2_score(y_test,y_pred))
    
    sns.scatterplot(x=y_test,y=y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.show()

    #code added by Kim
    #Model Dictionary =
    #Airport ID : [Model description, Lat, Lon, average IFR Value for airport, Model object, [Beta list], [Beta SE list], [pvalue list]]

    model_dict[key] = ['MLR1', data['LATITUDE'][0], data['LONGITUDE'][0],
               data['IFR'].mean(), linReg, [interc, coeff] ]
    


#code added by Kim
#Pickle dump to save model_dict to file
    
# create a binary pickle file 
f = open("model_dict.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(model_dict,f)

# close file
f.close()    


