#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:46:44 2022

@author: ajgray
"""

import pandas as pd

#testing code
date = '4/5/2021'
date_obj = pd.to_datetime(date) 

            
# Setting any output to display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def read_csv(csvfile,skiprows=0,header=0):
    '''

    Parameters
    ----------
    csvfile : string filepath
        this is the csv file to be read in.

    Returns
    -------
    takes as argument, the csvfile, and returns a raw pandas dataframe.

    '''
    
    return pd.read_csv(csvfile, skiprows=skiprows, header=header)

ops = read_csv('fairbanks.csv',6,0)
weather = read_csv('fairbanks_weather.csv')
holidays = read_csv('holiday_dates.csv')

# rename date variable
weather = weather.rename(columns={'DATE':'Date'})

weather['Date'] = pd.to_datetime(weather['Date']).dt.date
holidays['Date'] = pd.to_datetime(holidays['Date']).dt.date

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

def addLocID():
    '''
    This function uses a dictionary to match the airport to its respective airport code.
    It then adds a column to the merged dataset equal to the airport code.
    
    '''
    
    for key in dict1.keys():
        if key in ops_weather.loc[[1]]['NAME'].values[0]:
            for key, value in dict1[key].items():
                if key == 'LOC':
                    ops_weather[key] = value
                #if key == 'LAT':
                 #   ops_weather[key] = value 
               # if key == 'LONG':
                 #   ops_weather[key] = value
            print('\n')
            
def tagHolidays():
    '''
    This function loops through the ops_weather merged dates and tries to find matches in the
    holidays dates. 
    If there is a match, the ops_weather dataset is updated:
        1. A new coloumn (isAHoliday) is assigned the value 1, else zero. 
        2. A new column (Holiday) is assinged the appropriate holiday for that date

    '''
    
    holDates = []
    for i in range(len(holidays)):
        holDates.append(holidays.loc[i, 'Date'])
        
    for target_date in range(len(ops_weather)):
        if ops_weather.loc[target_date, 'Date'] not in holDates:
            #print(ops_weather.loc[target_date, 'Date'])
            ops_weather.loc[target_date, 'isAHoliday'] = 0
            ops_weather.loc[target_date, 'Holiday'] = 'None'
        else:
            for holiday_date in range(len(holidays)):
                if ops_weather.loc[target_date, 'Date'] == holidays.loc[holiday_date, 'Date']:
                    ops_weather.loc[target_date,'isAHoliday'] = 1
                    ops_weather.loc[target_date, 'Holiday'] = holidays.loc[holiday_date, 'Holiday']
                    #print('opsDate: {} equals holDate: {}\nHoliday: {}\n'.format(ops_weather.loc[target_date,'Date'],holidays.loc[holiday_date,'Date'],holidays.loc[holiday_date,'Holiday']))
                    #break
                #else:
                    #ops_weather.loc[target_date, 'isAHoliday'] = 0
                    #break
    
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
ops['Date'] = pd.to_datetime(ops['Date']).dt.date

# Selecting all numeric columns to change type to int64
numeric_columns = ops.columns[1:]

# Converting numeric columns from object to int64
ops[numeric_columns] = ops[numeric_columns].astype('int64')

# Creating variable for total IFR traffic
ops['IFR'] = ops['IFR Total'] + ops['IFR Overflight Total']

# Creating variable for total VFR traffic, which is target variable
ops['VFR'] = ops['VFR Total'] + ops['VFR Overflight Total'] + ops['Local Total']

#locallyDocked()


#grp1 = ops.groupby('Local Ratio',as_index=False)
#grp1_count = grp1[['Local Civil','Local Military','Local Total']].count()
#print(grp1_count)

# Check Dataframe
#print(ops.isna().sum())
#print(ops.dtypes)

#print(ops.info())
#print(ops.describe())

# merge ops and weather
ops_weather = ops.merge(weather,on='Date',how='inner') 

dict1 = {'FAIRBANKS INTERNATIONAL AIRPORT':{'LOC':'FAI','LAT':64.815,'LONG':-147.8563888889},
         'JUNEAU INTERNATIONAL AIRPORT':{'LOC':'JNU','LAT': 58.355, 'LONG':-134.57639},
         'ANCHORAGE INTERNATIONAL AIRPORT':{'LOC':'ANC','LAT':61.1741666667, 'LONG':-149.9816666667}
         }

# getting the text of the NAME column
ops_weather.loc[[1]]['NAME'].values[0] #OR
ops_weather['NAME'].head(1).values[0] # will give 'FAIRBANKS INTERNATIONAL AIRPORT, AK US'


addLocID()
# merged dataset columns
column_names = ['Date','LOC','STATION','NAME','LATITUDE','LONGITUDE','IFR Air Carrier','IFR Air Taxi', 'IFR General Aviation',
       'IFR Military', 'IFR Total', 'IFR Overflight Air Carrier',
       'IFR Overflight Air Taxi', 'IFR Overflight General Aviation',
       'IFR Overflight Military', 'IFR Overflight Total', 'VFR Air Carrier',
       'VFR Air Taxi', 'VFR General Aviation', 'VFR Military', 'VFR Total',
       'VFR Overflight Air Carrier', 'VFR Overflight Air Taxi',
       'VFR Overflight General Aviation', 'VFR Overflight Military',
       'VFR Overflight Total', 'Local Civil', 'Local Military', 'Local Total','IFR','VFR',
       'Total Airport Operations', 'Total Tower Operations','AWND', 'PRCP', 'SNOW', 'SNWD', 'TAVG', 'TMAX',
       'TMIN','isAHoliday','Holiday']

tagHolidays()

# set merged dataset columns to the new order
ops_weather = ops_weather.reindex(columns=column_names)

#print(ops_weather.head())

ops_weather.to_csv('data.csv')
