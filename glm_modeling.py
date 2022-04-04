#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:17:10 2022

@author: ajgray
"""

import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# This python script runs 3 GLM models on ACY, GFK, LGA, EWR, TEB, JFK, CVG
# All the code is preset for these 7. The code will output 3 fitted models 
# Along with 3 visuals for each fitted model
# The 3 models that are fitted on these 7 are:
# NegativeBinomial, GeneralizedPoission levels 1,2 and/or 3 (CVG)

# set defult seaborn theme
sns.set()

# Setting any output to display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read in model dictionary from disk

file_to_read = open("datasets.pkl", "rb")
datasets = pickle.load(file_to_read)
model_dict = dict()

# For DVT, VNY, use: family=sm.families.Gamma(link=sm.families.links.log()) <-- line 66 (poisson_training), random seed 1
# For PRC, use ONLY NegativeBinomial (comment out gp2model and gp1model)
data = datasets['ACY']#.loc[730:1094]


# Create day, month, and weekday predictor variables
for i in range(len(data)):
    data.loc[i, 'Month'] = data.loc[i, 'Date'].month
    data.loc[i, 'Day'] = data.loc[i, 'Date'].day
    data.loc[i, 'Week_Day'] = data.loc[i, 'Date'].weekday()

data.index = data.Date

# Create training and testing data sets
np.random.seed(7)
rand_selection = np.random.rand(len(data)) < .8
data_train = data[rand_selection]
data_test = data[~rand_selection]

# day and month are often not significant
#expr = """ VFR ~  Month + Day + Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """
expr = """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """


y_train, X_train = dmatrices(expr, data_train, return_type='dataframe')
y_test, X_test = dmatrices(expr, data_test, return_type='dataframe')
actual_counts = y_test['VFR']

def NegativeBinomial():
    # Training a Poisson regression model from the statsmodels GLM class
    poisson_training = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(link=sm.families.links.log()))
    results = poisson_training.fit()
    
    # Print training model
    print('\n\n')
    print(results.summary())
    
    # Checking predictions
    poisson_predictions = results.get_prediction(X_test)
    predictions_summary_frame = poisson_predictions.summary_frame()
    #print(predictions_summary_frame.head(15))
    
    
    predicted_counts = predictions_summary_frame['mean']
    visualizeModel(data, 'Negative Binomial',predicted_counts)
    
    return results

# [el for el in predictions_summary_frame.index if (el.year == 2017) and (el.month == 7)]
def generalizePoisson():
    
    gen_poisson_gp1 = sm.GeneralizedPoisson(y_train, X_train, p=1)
    
    # Fit (train) the model:
    results = gen_poisson_gp1.fit(method='newton')
    print(results.summary())

    gen_poisson_gp1_predictions = results.predict(X_test)
    predicted_counts=gen_poisson_gp1_predictions

    visualizeModel(data, "Consul's Generalized Poisson", predicted_counts)
    
    return results

def generalizedPoisson2():
    # Fit (train) the model:
    if data.LOC[0] == 'CVG':
        gen_poisson_gp2 = sm.GeneralizedPoisson(y_train, X_train, p=3)
    else:
        gen_poisson_gp2 = sm.GeneralizedPoisson(y_train, X_train, p=2)
        
    results = gen_poisson_gp2.fit(method='newton')
    print(results.summary())
    
    gen_poisson_gp2_predictions = results.predict(X_test)
    predicted_counts=gen_poisson_gp2_predictions
    
    visualizeModel(data, "Famoye’s Restricted Generalized Poisson",predicted_counts)
    
    return results
    

def visualizeModel(df, vis, predicted_counts):
 
    fig = plt.figure()
    fig.set_size_inches(15,8)
    fig.suptitle('{}: Predicted vs Actual VFR counts for {}'.format(vis,df.LOC[0]))
    predicted, = plt.plot(X_test.index, predicted_counts, 'go-', label='Predicted counts')
    actual, = plt.plot(X_test.index, actual_counts, 'ro-', alpha=.5, label='Actual counts')
    plt.xlabel('Date')
    plt.ylabel('VFR Counts')
    plt.legend(handles=[predicted, actual])
    plt.show()
    
    plt.clf()
    fig = plt.figure()
    fig.suptitle('{}: Scatter plot of Actual versus Predicted counts for {}'.format(vis,df.LOC[0]))
    plt.scatter(x=predicted_counts, y=actual_counts, marker='.')
    plt.xlabel('Predicted counts')
    plt.ylabel('Actual counts')
    plt.show()
    

nb = NegativeBinomial()
gp2model = generalizedPoisson2()
gp1model = generalizePoisson()



