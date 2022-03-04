#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:46:44 2022

@author: ajgray
"""
# code for getting each row's type for all non-Date columns of type object

#for col in ops.columns[1:]:
#    if header[col].dtype == 'O':
#        print('{}\n'.format(col))
#        for row in ops[col]:
#            if len(row) > 3:
#                print(row)
#        print('\n')

# Code for accessing each non-Date column and removing commas for values greater
# than 999


import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt


# Setting any output to display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def read_csv(csvfile):
    '''

    Parameters
    ----------
    csvfile : string filepath
        this is the csv file to be read in.

    Returns
    -------
    takes as argument, the csvfile, and returns a raw pandas dataframe.

    '''
    
    return pd.read_csv(csvfile, skiprows=6, header=0)

ops = read_csv('Juneau.csv')

# Drop last column, which is an empty column
ops.drop(ops.columns[-1], axis=1, inplace=True)

# Drop last 5 rows, which is unecessary footer text
ops = ops.iloc[:-5,:]

# Parse out and group individual columns for relabelling
ifrItinerant = ops.columns[1:6]
ifrOverflight = ops.columns[6:11]

vfrItinerant = ops.columns[11:16]
vfrOverflight = ops.columns[16:21]

local = ops.columns[21:24]
total = ops.columns[24:]

# Tagging IFR Itinerant columns
def tagFlightRules(colArray, rules):
    '''
        Add Flight Rules to applicable columns
        
    '''
    for i in range(colArray.shape[0]):
        if i == 0:
            colArray.values[i] = '{} Air Carrier'.format(rules)
        elif i == 1:
            colArray.values[i] = '{} Air Taxi'.format(rules)
        elif i == 2:
            colArray.values[i] = '{} General Aviation'.format(rules)
        elif i == 3:
            colArray.values[i] = '{} Military'.format(rules)
        else:
            colArray.values[i] = '{} Total'.format(rules)

def tagLocal():
    '''
        Tag Local variables        
    '''
    local.values[0] = 'Local Civil'
    local.values[1] = 'Local Military'
    local.values[2] = 'Local Total'

def tagTotal():
    '''
        Tag Total variables
    '''
    total.values[-2] = 'Total Airport Operations'
    total.values[-1] = 'Total Tower Operations'
    
def removeCommas():
    '''
    This function accesses the row of each non-Date column of the DataFrame, checks to see if
    that row's length is greater than 3 (i.e. 1,001), and modifies that row
    by removing the comma.

    '''
    
    for col in ops.columns[1:]:
        if ops[col].dtype == 'O':
            for i in range(1,len(ops[col])):
                if len(ops[col].loc[i]) > 3:
                    ops.loc[i, col] = ops[col].loc[i].replace(',','')

def locallyDocked():
    '''
    This function calculates daily local presence (Local Civil/ Local Total)
    NOTE: This function is required to take care of division by zero error

    Returns
    -------
    None.

    '''
    
    # Calculating presence of aircraft that are locally docked
    # Looping over Local Total column
    for i in range(1, ops['Local Total'].shape[0]+1):
        # Catching any 0/0 cases by determing when Local Total is 0 and setting
        # our new column Local Ratio equal to zero.
       if (ops.loc[i, 'Local Total'] == 0):
            ops.loc[i, 'Local Ratio'] = 0
       else:
           ops.loc[i, 'Local Ratio'] = round(ops.loc[i, 'Local Civil']/ops.loc[i, 'Local Total'],3)
    
tagFlightRules(ifrItinerant, 'IFR')
tagFlightRules(ifrOverflight, 'IFR Overflight')
tagFlightRules(vfrItinerant, 'VFR')
tagFlightRules(vfrOverflight, 'VFR Overflight')

tagLocal()
tagTotal()

removeCommas()

# Specifically dropping first row where data is irrelavant after formatting
ops = ops.dropna()

# Converting Date column to datetime object
ops['Date'] = pd.to_datetime(ops['Date'])

# Selecting all numeric columns to change type to int64
numeric_columns = ops.columns[1:]

# Converting numeric columns from object to int64
ops[numeric_columns] = ops[numeric_columns].astype('int64')

# Creating variable for total IFR traffic
ops['IFR'] = ops['IFR Total'] + ops['IFR Overflight Total']

# Creating variable for total VFR traffic, which is target variable
ops['VFR'] = ops['VFR Total'] + ops['VFR Overflight Total']

locallyDocked()


grp1 = ops.groupby('Local Ratio',as_index=False)
grp1_count = grp1[['Local Civil','Local Military','Local Total']].count()
#print(grp1_count)

# Check Dataframe
print(ops.isna().sum())
print(ops.dtypes)

#print(ops.info())
#print(ops.describe())

