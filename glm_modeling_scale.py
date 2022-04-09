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
actual_counts = ""

def dropEmptyCols(key):
    for col in datasets[key].columns[7:]:
        if max(datasets[key][col]) == 0:
            datasets[key] = datasets[key].drop(col,axis=1)

#for key in datasets:
for key in ['CRP']:
    datasets[key] = datasets[key].drop(['isAHoliday'], axis=1)
    dropEmptyCols(key)
    data = datasets[key]
    #print(data.LOC[0])
    
        
    # Create day, month, and weekday predictor variables
    for i in range(len(data)):
        data.loc[i, 'Month'] = data.loc[i, 'Date'].month
        #data.loc[i, 'Day'] = data.loc[i, 'Date'].day
        #data.loc[i, 'Week_Day'] = data.loc[i, 'Date'].weekday()
    
    data.index = data.Date

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
    
    mod_descript = 'NegativeBinomial'
    model_dict[key] = [mod_descript, data['LATITUDE'][0],
                       data['LONGITUDE'][0], data['IFR'].mean(), results,
                       expr, list(X_test)]
    
    return results

def Gamma():
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
    visualizeModel(data, 'Gamma',predicted_counts)
    
    
    mod_descript = 'Gamma'
    model_dict[key] = [mod_descript, data['LATITUDE'][0],
                       data['LONGITUDE'][0], data['IFR'].mean(), results,
                       expr, list(X_test)]
    
    return results


# [el for el in predictions_summary_frame.index if (el.year == 2017) and (el.month == 7)]
def generalizePoisson():
    global expr
    global y_train,X_train
    mod_descript = 'generalizePoisson'
    count=1
    #print(data.columns)
    # Fit (train) the model:
    if datasets[key].LOC[0] == 'CVG':
        gen_poisson_gp1 = sm.GeneralizedPoisson(y_train, X_train, p=2)
    else:
        gen_poisson_gp1 = sm.GeneralizedPoisson(y_train, X_train, p=1)
    
    # Fit (train) the model:
    for col in datasets[key].columns[7:]:
        if len(datasets[key][col].unique()) == 2 and datasets[key][col].unique()[0] == 0. and datasets[key][col].unique()[1] == 1.:
            print('{}: {}'.format(col,datasets[key][col].unique()))
            
    #print(datasets[key].describe())
    results = gen_poisson_gp1.fit(method='newton')
    

    gen_poisson_gp1_predictions = results.predict(X_test)
    predicted_counts=gen_poisson_gp1_predictions

    
    
    #print(model_dict[key][4].summary())
    highPreds = highPredsLst(key,results)
    if len(highPreds) == 0:
        model_dict[key] = [mod_descript, datasets[key]['LATITUDE'][0],
                           datasets[key]['LONGITUDE'][0], hasIFR(datasets[key]), gen_poisson_gp1.fit(method='newton'),
                           expr, list(X_test)]
        visualizeModel(datasets[key], "Consul's Generalized Poisson", predicted_counts)
        
    #print(highPreds)
    if len(highPreds) > 0:
        #print(len(highPreds))
        #print(data.columns)
        #print('here')
       #print(highPreds[0])
        datasets[key] = datasets[key].drop([col for col in highPreds],axis=1)
        #print('HERE!!!')
        #print(datasets[key].columns)
        expr = getExpr(key)
        #print(expr)
        createTrainTest(expr)
        highPreds = []
        results = ""
        count+=1
        generalizePoisson()
        #print('LAST 1',results.summary())
     
    if count == 2:
        print('{}\n{}'.format(key,model_dict[key][4].summary()))
    
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

    
    mod_descript = 'generalizedPoisson2'
    model_dict[key] = [mod_descript, data['LATITUDE'][0],
                       data['LONGITUDE'][0], data['IFR'].mean(), results,
                       expr, list(X_test)]
    
    
    return results

def getExpr(key):
    dCols = list(datasets[key].columns[7:])
    #print(dCols)
    expr = "VFR ~ "
    
    for i in range(len(dCols)):
        if i == 0:
            expr+=dCols[i]
        else:
            expr+=' + {}'.format(dCols[i])
    print(expr)
    return expr

def createTrainTest(expr=""):
    global y_train,X_train,y_test,X_test,actual_counts
    # Create training and testing data sets
    np.random.seed(1)
    rand_selection = np.random.rand(len(datasets[key])) < .8
    data_train = datasets[key][rand_selection]
    data_test = datasets[key][~rand_selection]
    #print('CREATE TRT',expr)
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
    #sns.set(rc={"figure.figsize":(10, 5)})
# =============================================================================
#     sns.set_style('white')
#     sns.jointplot(x=predicted_counts, 
#               y=actual_counts, 
#               data=df,
#               kind="reg",
#               color='b',
#               marginal_kws=dict(bins=100, color='r'))
#     plt.savefig("marginal_plot_different_color_histogram_Seaborn.png", dpi=150)  
# =============================================================================
    
#assignment dictionary holds model and predictor selection decisions for a=each airport
    
# =============================================================================
# assign_dict = dict()
# assign_dict['ACY'] = [generalizePoisson, """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['GFK'] = [generalizePoisson, """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['LGA'] = [generalizePoisson, """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['EWR'] = [generalizePoisson, """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['TEB'] = [generalizePoisson,""" VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['JFK'] = [generalizedPoisson2,""" VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['CVG'] = [generalizedPoisson2, """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['PRC'] = [NegativeBinomial, """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['DVT'] = [Gamma, """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# assign_dict['VNY'] = [Gamma, """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """]
# =============================================================================



        
    #pull formula for airport from assignment dictionary
    #expr = assign_dict[key][1]
#for key in datasets:
for key in ['CRP']: 
    expr = getExpr(key)
    #print(expr)
    expr_dict[key] = expr
    # day and month are often not significant
    #expr = """ VFR ~  Month + Day + Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """
    #expr = """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """
    
    #expr = """ VFR ~ Week_Day + IFR + AWND + PRCP + PRCP_SQRT  + SNOW_SQRT + TMIN + TMAX """
    
    createTrainTest(getExpr(key))
        
    
    #call the assigned model for airport found in the assigned model dictionary
    
    #assign_dict[key][0]()
    if key == 'DVT':
        Gamma()
    else:
        generalizePoisson()
       # model.conf_int() - confidence intervals
       # conf_ints = dict()
       # for i in range(len(model.conf_int())):
           #conf_ints[model.conf_int().iloc[[i]].index[0]] = model.conf_int().iloc[i].values
       # pvals = dict()
       # for i in range(len(model.pvalues)):
           #pvals[model.pvalues.iloc[[i]].index[0]] = model.pvalues.iloc[i]
       # model.pvalues
       # model.params
       #model.llnull
       # model.llf
       # model.llr_pvalue
       # model.prsquared
       
       # remove non-significant columns
       # for col in pvals:
          # if col != 'alpha':
           #    if pvals[col] > 0.05:
            #       datasets['ABI'] = datasets['ABI'].drop(col,axis=1)
        




#Pickle dump to save model_dict to file
        
# create a binary pickle file 
f = open("model_dict.pkl","wb")
    
# write the python object (dict) to pickle file
pickle.dump(model_dict,f)
    
# close file
f.close()    



#nb = NegativeBinomial()
#gp2model = generalizedPoisson2()
#gp1model = generalizePoisson()
