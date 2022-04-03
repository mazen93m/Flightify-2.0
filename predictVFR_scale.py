#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:39:25 2022

@author: ajgray
"""

#Kim Cawi edited/added Code


#import dataCleaner as d
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import math

import pickle
import numpy as np

import statsmodels.api as sm
from patsy import dmatrices

#from sklearn import linear_model
#from sklearn.linear_model import PoissonRegressor

import statsmodels.formula.api as smf



#Model ID:  MLR1  Multiple Linear Regression 1



#Open pickle file on current directory for the datasets dictionary from the Data Cleaner
file_to_read = open("datasets.pkl", "rb")
datasets = pickle.load(file_to_read)
file_to_read.close()

#start model_dict
model_dict = {}

#for key in datasets:
#for key in ["FAI", "ANC", "JNU"]: 
for key in ["JFK", "EWR", "LGA", "TEB", "VNY", "ACY", "GFK", "DVT", "PRC", "CVG"]:    
    
    #reading in the data csv file outputed by the cleaning tool
    #data = d.datasets['ANC']
    data = datasets[key]
    print(data['LOC'][0])
 
    
    sns.set()
    correlation_matrix = data[['VFR','AWND','SNOW','TMAX','isAHoliday','PRCP']].corr()
    
    sns.heatmap(correlation_matrix)
    plt.show()
    
    #histogram for VFR
    data['VFR'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
    axis=plt.title('VFR Counts for ' + key)
    plt.xlabel('Counts')
    plt.ylabel('VFR Counts')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    #scatterplot  target on y axis
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
   
    X = data_np[:, [-9,-8,-6,-3,-2,-1]]
    
    # dependent label variable
    y = data_np[:, -10]
    
    #y= y + .01
    #lny = np.log(y+1)
    
    predictor_names = ['IFR', 'AWND', 'PRCP_SQRT', 'TMAX', 'isAHoliday', 'SNOW_SQRT']
    #X = pd.DataFrame(data, columns = predictor_names)
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
    test_root_MSE = math.sqrt(metrics.mean_squared_error(y_test,y_pred))
    print('Test Root Mean squared error (MSE): %.2f'% math.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    R2 = r2_score(y_test,y_pred)
    print('Coefficient of determination (R^2): %.2f'% r2_score(y_test,y_pred))
    
    sns.scatterplot(x=y_test,y=y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    plt.show()

    
    #Model Dictionary =
    #Airport ID : [Model description, Lat, Lon, average IFR Value for airport,
    #              Model object, [intercept, [coeff list]], predictor_names,
    #             test_root_MSE, R squared, [Beta Standard error list],
    #             [pvalue list for intercept and coefficeints]]

    model_dict[key] = ['MLR1', data['LATITUDE'][0], data['LONGITUDE'][0],
               data['IFR'].mean(), linReg, [interc, coeff], predictor_names,
               test_root_MSE, R2 ]
    


#code added by Kim
#Pickle dump to save model_dict to file
    
# create a binary pickle file 
f = open("model_dict.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(model_dict,f)

# close file
f.close()    




# =============================================================================
# #Model ID: POIS1  Poisson Regressor
# 
# 
#     #sklearn PoissonRegressor
#    
#     poisReg = sklearn.linear_model.PoissonRegressor()
#     
#     #fit poisReg to the training set
#     poisReg.fit(X_train, y_train)
#     
#     poisReg.score(X, y)
# 
#     poisReg.coef_
# 
#     poisReg.intercept_
# 
#     y_pred = poisReg.predict(X_test)
#      
#     
#     #statsmodels version
#     X_train = sm.tools.tools.add_constant(X_train, prepend=True, has_constant='skip')
#     X_test = sm.tools.tools.add_constant(X_test, prepend=True, has_constant='skip')
# 
#     
#     poissReg = sm.discrete.discrete_model.Poisson()
#     poissReg.fit(y_train, X_train)
#     
#     #-----------
#     
# 
# 
#     
#     poissReg = sm.poisson('VFR = IFR + PRCP_SRT + SNOW + TMAX + isAHOLIDAY', data = datasets[key])
#     
#       
#     #predict(params[, exog, exposure, offset, linear])
#     y_pred = poisReg.predict(X_test)
#     
#     
#
# 
# #-----------------------------------
# =============================================================================




