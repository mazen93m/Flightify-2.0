#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:08:52 2022

@author: ajgray
"""

import pandas as pd


# Setting any output to display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


ops = pd.read_csv('fairbanks.csv', skiprows=6, header=0)

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
    
tagFlightRules(ifrItinerant, 'IFR')
tagFlightRules(ifrOverflight, 'IFR Overflight')
tagFlightRules(vfrItinerant, 'VFR')
tagFlightRules(vfrOverflight, 'VFR Overflight')

tagLocal()
tagTotal()
#renameColumns()

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

# Calculating presence of aircraft  that locally docked
ops['Local Ratio'] = round(ops['Local Civil']/ops['Local Total'],2)

# Check Dataframe
#print(ops.info())
#print(ops.describe())
