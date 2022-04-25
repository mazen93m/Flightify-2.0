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
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# This python script runs 3 GLM models on ACY, GFK, LGA, EWR, TEB, JFK, CVG
# All the code is preset for these 7. The code will output 3 fitted models 
# Along with 6 visuals for each fitted model
# The 3 models that are fitted on these 7 are:
# NegativeBinomial, GeneralizedPoission levels 1,2 and/or 3 (CVG)

# For DVT, VNY, use: family=sm.families.Gamma(link=sm.families.links.log()) <-- line 68 (poisson_training), random seed 1
# For PRC, use ONLY NegativeBinomial (comment out gp2model and gp1model at the bottom of the script)

# set defult seaborn theme
sns.set()

# Setting any output to display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read in model dictionary from disk

file_to_read = open("datasets.pkl", "rb")
datasets = pickle.load(file_to_read)
file_to_read.close()
model_dict = dict()
expr_dict = dict()

# Create training and testing data sets
y_train = ""
X_train = ""
y_test = ""
X_test = ""
expr = ""
actual_counts = ""
lst1 = []

def dropEmptyCols(key):
    for col in datasets[key].columns[8:]:
        if max(datasets[key][col]) == 0:
            datasets[key] = datasets[key].drop(col,axis=1)
            
        # Taking care of singular matrix error for glm poisson
        elif len(datasets[key][col].unique()) == 2: #and datasets[key][col].unique()[0] == 0. and datasets[key][col].unique()[1] == 1.:
            #print('{}: {}'.format(col,datasets[key][col].unique()))
            datasets[key] = datasets[key].drop(col,axis=1)


def findNull_llf():
    for key in model_dict:
        if pd.isnull(model_dict[key][4].llf):
            lst1.append(key)
            
    return lst1


def dropEmpties():
    for key in datasets:
        dropEmptyCols(key) 
  

def hasIFR(df):
    if 'IFR' in df.columns:
        return df['IFR'].mean()
    return 0

# =============================================================================
# def highPredictors(key):
#     pvals = dict()
#     highpvals = []
#     hasHighPreds = False
#     for i in range(len(model.pvalues)):
#         pvals[model.pvalues.iloc[[i]].index[0]] = model.pvalues.iloc[i]
#     
#     for col in pvals:
#         if col != 'alpha':
#             if pvals[col] > 0.05:
#                 highpvals.append(col)
#                 hasHighPreds = True
#                 
#     return hasHighPreds
# =============================================================================
    
def highPredsLst(key,results):
    pvals = dict()
    highpvals = []
    for i in range(1,len(results.pvalues)):
        pvals[results.pvalues.iloc[[i]].index[0]] = results.pvalues.iloc[i]
    
    for col in pvals:
        if col != 'alpha':
            if pvals[col] > 0.05:
                highpvals.append(col)
    
    return highpvals

def NegativeBinomial(key):
    global expr
    global y_train,X_train
    mod_descript = 'NegativeBinomial'
    
    # Training a Poisson regression model from the statsmodels GLM class
    nb_training = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(link=sm.families.links.log()))
    results = nb_training.fit()
    
    
    # Checking predictions
    nb_predictions = results.get_prediction(X_test)
    predictions_summary_frame = nb_predictions.summary_frame()

    predicted_counts = predictions_summary_frame['mean']
    
    highPreds = highPredsLst(key,results)
    if len(highPreds) == 0 or len(datasets[key].columns.values) == 8:
        y_pred = results.predict(X_test)
        rmse = round(math.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
        
        y_test_array = np.array(y_test)
        y_pred_array = np.array(y_pred)
        sum_errs = np.sum((y_test_array - y_pred_array)**2)
        stdev = round(np.sqrt(1 / (len(y_test_array) - 2) * sum_errs),2)


        
        model_dict[key] = [mod_descript, datasets[key]['LATITUDE'][0],
                           datasets[key]['LONGITUDE'][0], hasIFR(datasets[key]),  nb_training.fit(),
                           expr, list(X_test),rmse,stdev, nb_training, datasets[key]['Region'][0]] 
        
        print('{}\n{}: {}'.format(key,'RMSE',rmse))
        print('{}: {}'.format('STDEV',stdev))
        print('\n{}'.format(model_dict[key][4].summary()))

        visualizeModel(datasets[key], "Negative Binomial", predicted_counts)
        
        
    if len(highPreds) > 0 and len(datasets[key].columns[8:].values) > 1:
        datasets[key] = datasets[key].drop([col for col in highPreds],axis=1)
        expr = getExpr(key)
        createTrainTest(key,expr)
        highPreds = []
        results = ""
        NegativeBinomial(key)
    
    return results

def Gamma(key):
    # Training a Poisson regression model from the statsmodels GLM class
    poisson_training = sm.GLM(y_train, X_train, family=sm.families.Gamma(link=sm.families.links.log()))
    results = poisson_training.fit()
    
    # Print training model
    print('\n\n')
    print(results.summary())
    
    # Checking predictions
    poisson_predictions = results.get_prediction(X_test)
    predictions_summary_frame = poisson_predictions.summary_frame()
    #print(predictions_summary_frame.head(15))
    
    
    predicted_counts = predictions_summary_frame['mean']
    visualizeModel(datasets[key], 'Gamma',predicted_counts)
    
    
    mod_descript = 'Gamma'
    model_dict[key] = [mod_descript, datasets[key]['LATITUDE'][0],
                       datasets[key]['LONGITUDE'][0], datasets[key]['IFR'].mean(), results,
                       expr, list(X_test)]
    
    return results


def generalizePoisson(key):
    global expr
    global y_train,X_train
    mod_descript = 'generalizePoisson'
    
    # Fit (train) the model:
    if datasets[key].LOC[0] == 'CVG':
        gen_poisson_gp1 = sm.GeneralizedPoisson(y_train, X_train, p=2)
    else:
        gen_poisson_gp1 = sm.GeneralizedPoisson(y_train, X_train, p=1)
            
    results = gen_poisson_gp1.fit(method='newton')

    if pd.isnull(results.llf):
        if 'SNOW' in datasets[key].columns:
            datasets[key] = datasets[key].drop('SNOW',axis=1)
            expr = getExpr(key)
            createTrainTest(key,expr)
            generalizePoisson(key)

        elif 'SNOW_SQRT' in datasets[key].columns:
            datasets[key] = datasets[key].drop('SNOW_SQRT',axis=1)
            expr = getExpr(key)
            createTrainTest(key,expr)
            generalizePoisson(key)           
            
    gen_poisson_gp1_predictions = results.predict(X_test)
    predicted_counts=gen_poisson_gp1_predictions

    
    
    highPreds = highPredsLst(key,results)
    if len(highPreds) == 0 and pd.isnull(results.llf) == False:
        y_pred = results.predict(X_test)
        rmse = round(math.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
        
        y_test_array = np.array(y_test)
        y_pred_array = np.array(y_pred)
        sum_errs = np.sum((y_test_array - y_pred_array)**2)
        stdev = round(np.sqrt(1 / (len(y_test_array) - 2) * sum_errs),2)


        
        model_dict[key] = [mod_descript, datasets[key]['LATITUDE'][0],
                           datasets[key]['LONGITUDE'][0], hasIFR(datasets[key]), gen_poisson_gp1.fit(method='newton'),
                           expr, list(X_test),rmse,stdev,gen_poisson_gp1, datasets[key]['Region'][0]]
        
        print('{}\n{}: {}'.format(key,'RMSE',rmse))
        print('{}: {}'.format('STDEV',stdev))
        print('\n{}'.format(model_dict[key][4].summary()))

        visualizeModel(datasets[key], "Consul's Generalized Poisson", predicted_counts)
        
        
    if len(highPreds) > 0:
        datasets[key] = datasets[key].drop([col for col in highPreds],axis=1)
        expr = getExpr(key)
        createTrainTest(key,expr)
        highPreds = []
        results = ""
        generalizePoisson(key)

    return results

def generalizedPoisson2(key):
    # Fit (train) the model:
    if datasets[key].LOC[0] == 'CVG':
        gen_poisson_gp2 = sm.GeneralizedPoisson(y_train, X_train, p=3)
    else:
        gen_poisson_gp2 = sm.GeneralizedPoisson(y_train, X_train, p=2)
        
    results = gen_poisson_gp2.fit(method='newton')
    print(results.summary())
    
    gen_poisson_gp2_predictions = results.predict(X_test)
    predicted_counts=gen_poisson_gp2_predictions
    
    visualizeModel(datasets[key], "Famoye’s Restricted Generalized Poisson",predicted_counts)

    
    mod_descript = 'generalizedPoisson2'
    model_dict[key] = [mod_descript, datasets[key]['LATITUDE'][0],
                       datasets[key]['LONGITUDE'][0], datasets[key]['IFR'].mean(), results,
                       expr, list(X_test)]
    
    
    return results

def getExpr(key):
    dCols = list(datasets[key].columns[8:])
    #print(dCols)
    expr = "VFR ~ "
    
    for i in range(len(dCols)):
        if i == 0:
            expr+=dCols[i]
        else:
            expr+=' + {}'.format(dCols[i])
    #print(expr)
    return expr

def createTrainTest(key, expr=""):
    global y_train,X_train,y_test,X_test,actual_counts
    # Create training and testing data sets
    np.random.seed(1)
    rand_selection = np.random.rand(len(datasets[key])) < .8
    data_train = datasets[key][rand_selection]
    data_test = datasets[key][~rand_selection]
    y_train, X_train = dmatrices(expr, data_train, return_type='dataframe')
    y_test, X_test = dmatrices(expr, data_test, return_type='dataframe')
    actual_counts = y_test['VFR']

def visualizeModel(df, vis, predicted_counts):

# =============================================================================
#     fig = plt.figure()
#     fig.set_size_inches(25,10)
#     fig.suptitle('{}: Predicted vs Actual VFR counts for {}'.format(vis,df.LOC[0]), fontsize=25)
#     predicted, = plt.plot(X_test.index, predicted_counts, 'go-', label='Predicted counts')
#     actual, = plt.plot(X_test.index, actual_counts, 'ro-', alpha=.5, label='Actual counts')
#     plt.xlabel('Year', fontsize= 20)
#     plt.ylabel('VFR Counts', fontsize=20)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.legend(handles=[predicted, actual])
#     plt.show()
# =============================================================================
    
    sns.set_style('white')
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(20,10)
    fig.suptitle('{}: Scatter plot of Actual versus Predicted counts for {}'.format(vis,df.LOC[0]), fontsize=25)
    plt.scatter(x=predicted_counts, y=actual_counts, marker='.')
    sns.regplot(x=predicted_counts, y=actual_counts, data=df)
    plt.xlabel('Predicted Counts', fontsize=20)
    plt.ylabel('Actual Counts', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    
    plt.clf()
    
def runModels():
    dropEmpties()
    for key in datasets:
        if key in ['SBD','IFP']:
            continue
        expr = getExpr(key)
        createTrainTest(key,expr)
        if key == 'AEX':
            NegativeBinomial(key)
        else:
            generalizePoisson(key)
 
# Run the program 
runModels()


#Pickle dump to save model_dict to file
        
# create a binary pickle file 
f = open("model_dict.pkl","wb")
    
# write the python object (dict) to pickle file
pickle.dump(model_dict,f)
    
# close file
f.close()    
