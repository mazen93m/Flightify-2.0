#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:46:44 2022

@author: ajgray
"""

import pandas as pd
import numpy as np
import airports as a


# Setting any output to display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

airports = a.airports_dict
dataFrameLst = []
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

def create_dataFrames():
    
    iterator = iter(airports)
    global dataFrameLst
    
    for i in range(len(airports)):
        nxt = next(iterator)
        #print(nxt)
        dataFrameLst+=[[read_csv(airports[nxt][0],6,0),read_csv(airports[nxt][1])]]
        if i == len(airports)-1:
            #print(i)
            break
    return dataFrameLst

create_dataFrames()
holidays = read_csv('holiday_dates.csv')

# =============================================================================
# ops = read_csv('anc_opsnet_tower_ops_2017-2021.csv',6,0)
# weather = read_csv('anc_noaa_usw00026451_2017-2021.csv')
# =============================================================================

def renameDate():
    for el in dataFrameLst:
        el[1] = el[1].rename(columns={'DATE':'Date'})


def to_datetime():
    for el in dfLst:
        # ops data Date column
        el[0]['Date'] = pd.to_datetime(el[0]['Date']).dt.date
        # weather data Date column
        el[1]['Date'] = pd.to_datetime(el[1]['Date']).dt.date
        #el[1]['Dat']
        #print(el[1]['Date'].dtype)
    holidays['Date'] = pd.to_datetime(holidays['Date']).dt.date

def dropLastCol():
    for el in dataFrameLst:
        el[0].drop(el[0].columns[-1], axis=1, inplace=True)

def dropLast5Rows():
    for el in dataFrameLst:
        el[0] = el[0].iloc[:-5,:]
    
renameDate()
dropLastCol()
dropLast5Rows()

# REMOVE LINES
# rename date variable
#weather = weather.rename(columns={'DATE':'Date'})
#weather['Date'] = pd.to_datetime(weather['Date']).dt.date

# rREMOVE LINE
#holidays['Date'] = pd.to_datetime(holidays['Date']).dt.date

# REMOVE LINE
# Drop last column, which is an empty column
#ops.drop(ops.columns[-1], axis=1, inplace=True)

# REMOVE LINE
# Drop last 5 rows, which is unecessary footer text
#ops = ops.iloc[:-5,:]

def ifrItinerant_Lst():
    #print(ifrItinerant2,'n\n\n')
    ifrCleaned = tagFlightRulesLst(ifrItinerant2, 'IFR')
    #print(ifrCleaned)
    first_El = [[df[0]] for df in dataFrameLst]
    second_El = [[df[1]] for df in dataFrameLst]

    #print(len(first_El))
    for el in first_El:
        #print(el[0].head(1))
        for i in range(5):
            el[0] = el[0].rename(columns={el[0].columns[1:6][i]:ifrCleaned[i]})
        #print(el[0].columns[1:6])
    dataFrameLst2 = []
    for i in range(len(first_El)):
        #print(type(first_El[i]))
        #print(type(second_El[i]))

        dataFrameLst2+=[[first_El[i][0], second_El[i][0]]]
    return dataFrameLst2

def ifrOverflight_Lst(dfLst):
    ifrOvCleaned = tagFlightRulesLst(ifrOverflight2, 'IFR Overflight')
    first_El = [[df[0]] for df in dfLst]
    second_El = [[df[1]] for df in dfLst]

    for el in first_El:
        #print(el[0].head(1))
        for i in range(5):
            el[0] = el[0].rename(columns={el[0].columns[6:11][i]:ifrOvCleaned[i]})

    dataFrameLst2 = []
    for i in range(len(first_El)):
        dataFrameLst2+=[[first_El[i][0], second_El[i][0]]]
        
    return dataFrameLst2

def vfr_Lst(dfLst):
    vfrCleaned = tagFlightRulesLst(vfrItinerant2, 'VFR')
    first_El = [[df[0]] for df in dfLst]
    second_El = [[df[1]] for df in dfLst]

    for el in first_El:
        for i in range(5):
            el[0] = el[0].rename(columns={el[0].columns[11:16][i]:vfrCleaned[i]})

    dataFrameLst2 = []
    for i in range(len(first_El)):
        dataFrameLst2+=[[first_El[i][0], second_El[i][0]]]
        
    return dataFrameLst2

def vfrOverflight_Lst(dfLst):
    vfrOvCleaned = tagFlightRulesLst(vfrOverflight2, 'VFR Overflight')
    first_El = [[df[0]] for df in dfLst]
    second_El = [[df[1]] for df in dfLst]

    for el in first_El:
        for i in range(5):
            el[0] = el[0].rename(columns={el[0].columns[16:21][i]:vfrOvCleaned[i]})

    dataFrameLst2 = []
    for i in range(len(first_El)):
        dataFrameLst2+=[[first_El[i][0], second_El[i][0]]]
        
    return dataFrameLst2

def local_Lst(dfLst):
    localCleaned = tagLocal2()
    first_El = [[df[0]] for df in dfLst]
    second_El = [[df[1]] for df in dfLst]

    for el in first_El:
        for i in range(3):
            el[0] = el[0].rename(columns={el[0].columns[21:24][i]:localCleaned[i]})

    dataFrameLst2 = []
    for i in range(len(first_El)):
        dataFrameLst2+=[[first_El[i][0], second_El[i][0]]]
        
    return dataFrameLst2

def total_Lst(dfLst):
    totalCleaned = tagTotal2()
    first_El = [[df[0]] for df in dfLst]
    second_El = [[df[1]] for df in dfLst]

    for el in first_El:
        for i in range(2):
            #print()
            el[0] = el[0].rename(columns={el[0].columns[24:][i]:totalCleaned[i]})

    dataFrameLst2 = []
    for i in range(len(first_El)):
        dataFrameLst2+=[[first_El[i][0], second_El[i][0]]]
        
    return dataFrameLst2

def tagColumns():
    
    # tagging IFR for the itinerant columns
    dfLst_ifr = ifrItinerant_Lst()
    dfLst_ifr_ovfl = ifrOverflight_Lst(dfLst_ifr)
    dfLst_ifr_ovfl_vfr = vfr_Lst(dfLst_ifr_ovfl)
    dfLst_ifr_ovfl_vfr_ovfl = vfrOverflight_Lst(dfLst_ifr_ovfl_vfr)
    dfLst_ifr_ovfl_vfr_ovfl_local = local_Lst(dfLst_ifr_ovfl_vfr_ovfl)
    
    dfLst = total_Lst(dfLst_ifr_ovfl_vfr_ovfl_local)
    
    return dfLst

def delCol(*cols):
    for col in cols:
        for dataset in merged_Lst:
            #print(type(col))
            if col in [col for dataset in merged_Lst for col in dataset.columns]:
                del dataset[col]
    
# Parse out and group individual columns for relabelling
ifrItinerant2 = dataFrameLst[0][0].columns[1:6]
ifrOverflight2 = dataFrameLst[0][0].columns[6:11]

vfrItinerant2 = dataFrameLst[0][0].columns[11:16]
vfrOverflight2 = dataFrameLst[0][0].columns[16:21]

local2 = dataFrameLst[0][0].columns[21:24]
total2 = dataFrameLst[0][0].columns[24:]

# =============================================================================
# ifrItinerant = ops.columns[1:6]
# ifrOverflight = ops.columns[6:11]
# 
# vfrItinerant = ops.columns[11:16]
# vfrOverflight = ops.columns[16:21]
# 
# local = ops.columns[21:24]
# total = ops.columns[24:]
# =============================================================================

# merged dataset columns
column_names = ['Date','LOC','STATION','NAME','LATITUDE','LONGITUDE','IFR Air Carrier','IFR Air Taxi', 'IFR General Aviation',
       'IFR Military', 'IFR Total', 'IFR Overflight Air Carrier',
       'IFR Overflight Air Taxi', 'IFR Overflight General Aviation',
       'IFR Overflight Military', 'IFR Overflight Total', 'VFR Air Carrier',
       'VFR Air Taxi', 'VFR General Aviation', 'VFR Military', 'VFR Total',
       'VFR Overflight Air Carrier', 'VFR Overflight Air Taxi',
       'VFR Overflight General Aviation', 'VFR Overflight Military',
       'VFR Overflight Total', 'Local Civil', 'Local Military', 'Local Total', 'Total Airport Operations', 
       'Total Tower Operations','VFR','IFR','AWND', 'PRCP', 'PRCP_SQRT', 'SNOW', 'TMIN','TMAX',
       'isAHoliday']

# Tagging IFR Itinerant columns
def tagFlightRulesLst(colArray, rules):
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
            
    return colArray


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

def tagLocal2():
     '''
         Tag Local variables        
     '''
     local2.values[0] = 'Local Civil'
     local2.values[1] = 'Local Military'
     local2.values[2] = 'Local Total'
     
     return local2
 
# =============================================================================
# def tagLocal():
#     '''
#         Tag Local variables        
#     '''
#     local.values[0] = 'Local Civil'
#     local.values[1] = 'Local Military'
#     local.values[2] = 'Local Total'
#     
# def tagTotal():
#     '''
#         Tag Total variables
#     '''
#     total.values[-2] = 'Total Airport Operations'
#     total.values[-1] = 'Total Tower Operations'
# 
# =============================================================================
def tagTotal2():
    '''
        Tag Total variables
    '''
    total2.values[-2] = 'Total Airport Operations'
    total2.values[-1] = 'Total Tower Operations'
    
    return total2

dfLst = tagColumns()

# =============================================================================
# def removeCommas():
#     '''
#     This function accesses the row of each non-Date column of the DataFrame, checks to see if
#     that row's length is greater than 3 (i.e. 1,001), and modifies that row
#     by removing the comma.
# 
#     '''
#     
#     for col in ops.columns[1:]:
#         if ops[col].dtype == 'O':
#             for i in range(1,len(ops[col])):
#                 if len(ops[col].loc[i]) > 3:
#                     ops.loc[i, col] = ops[col].loc[i].replace(',','')
# 
# =============================================================================

def removeCommas2():
    '''
    This function accesses the row of each non-Date column of the DataFrame, checks to see if
    that row's length is greater than 3 (i.e. 1,001), and modifies that row
    by removing the comma.

    '''
    
    for el in dfLst:
        for col in el[0].columns:
            if col.lower() != 'date' and el[0][col].dtype == 'O':
                for i in range(1,len(el[0][col])):
                    if len(el[0][col].loc[i]) > 3:
                        el[0].loc[i, col] = el[0][col].loc[i].replace(',','')

                
def dropNA():
    for el in dfLst:
        el[0] = el[0].dropna()


# =============================================================================
# def addLocID():
#     '''
#     This function uses a dictionary to match the airport to its respective airport code.
#     It then adds a column to the merged dataset equal to the airport code.
#     
#     '''
#     
#     for key in dict1.keys():
#         if key in ops_weather.loc[[1]]['NAME'].values[0]:
#             for key, value in dict1[key].items():
#                 if key == 'LOC':
#                     ops_weather[key] = value
#                 #if key == 'LAT':
#                  #   ops_weather[key] = value 
#                # if key == 'LONG':
#                  #   ops_weather[key] = value
#             print('\n')
#             
# =============================================================================
def addLocIDLst():
    '''
    This function uses a dictionary to match the airport to its respective airport code.
    It then adds a column to the merged dataset equal to the airport code.
    
    '''
    
    for key in airports.keys():
        for dataset in merged_Lst:
            if airports[key][2].lower() == dataset.loc[[0]]['NAME'].values[0].lower():
                dataset['LOC'] = key.upper()
                #if key == 'LAT':
                 #   ops_weather[key] = value 
               # if key == 'LONG':
                 #   ops_weather[key] = value
            
# =============================================================================
# def tagHolidays():
#     '''
#     This function loops through the ops_weather merged dates and tries to find matches in the
#     holidays dates. 
#     If there is a match, the ops_weather dataset is updated:
#         1. A new coloumn (isAHoliday) is assigned the value 1, else zero. 
#         2. A new column (Holiday) is assinged the appropriate holiday for that date
# 
#     '''
#     
#     holDates = []
#     for i in range(len(holidays)):
#         holDates.append(holidays.loc[i, 'Date'])
#         
#     for target_date in range(len(ops_weather)):
#         if ops_weather.loc[target_date, 'Date'] not in holDates:
#             #print(ops_weather.loc[target_date, 'Date'])
#             ops_weather.loc[target_date, 'isAHoliday'] = 0
#             ops_weather.loc[target_date, 'Holiday'] = 'None'
#         else:
#             for holiday_date in range(len(holidays)):
#                 if ops_weather.loc[target_date, 'Date'] == holidays.loc[holiday_date, 'Date']:
#                     ops_weather.loc[target_date,'isAHoliday'] = 1
#                     ops_weather.loc[target_date, 'Holiday'] = holidays.loc[holiday_date, 'Holiday']
#                     #print('opsDate: {} equals holDate: {}\nHoliday: {}\n'.format(ops_weather.loc[target_date,'Date'],holidays.loc[holiday_date,'Date'],holidays.loc[holiday_date,'Holiday']))
# =============================================================================
def tagHolidaysLst():
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
    for el in merged_Lst:
        
        for target_date in range(len(el)):
            if el.loc[target_date, 'Date'] not in holDates:
                #print(ops_weather.loc[target_date, 'Date'])
                el.loc[target_date, 'isAHoliday'] = 0
                #el.loc[target_date, 'Holiday'] = 'None'
            else:
                for holiday_date in range(len(holidays)):
                    if el.loc[target_date, 'Date'] == holidays.loc[holiday_date, 'Date']:
                        el.loc[target_date,'isAHoliday'] = 1
                        #el.loc[target_date, 'Holiday'] = holidays.loc[holiday_date, 'Holiday']
                        #print('opsDate: {} equals holDate: {}\nHoliday: {}\n'.format(ops_weather.loc[target_date,'Date'],holidays.loc[holiday_date,'Date'],holidays.loc[holiday_date,'Holiday']))
def to_int():
    for el in dfLst:
        el[0][numeric_columns2] = el[0][numeric_columns2].astype('int64')


def vfr_ifr():
    for el in dfLst:
        el[0]['VFR'] = el[0]['VFR Total'] + el[0]['VFR Overflight Total']
        el[0]['IFR'] = el[0]['IFR Total'] + el[0]['IFR Overflight Total']
        
def PRCP_SQRT():
    for el in dfLst:
        el[1]['PRCP_SQRT'] = el[1]['PRCP']**(1./2)
         
def merge_datasets():
    merged_Lst = []
    for el in dfLst:
        merged_dataset = el[0].merge(el[1],on='Date',how='inner')
        merged_Lst+=[merged_dataset]
        
    return merged_Lst

def setCols():
    # set merged dataset columns an ordered subset list of columns
    for i in range(len(merged_Lst)):
        merged_Lst[i] = merged_Lst[i].reindex(columns=column_names)

# save data to a dictionary
def create_dataDict():
    
    keys = [merged_Lst[i]['LOC'][0] for i in range(len(merged_Lst))]
    values = [merged_Lst[i] for i in range(len(merged_Lst))]
    datasets = {}
    
    for i in range(len(merged_Lst)):
        datasets[keys[i]] = values[i]
    
    return datasets
        
# =============================================================================
# tagFlightRules(ifrItinerant, 'IFR')
# tagFlightRules(ifrOverflight, 'IFR Overflight')
# tagFlightRules(vfrItinerant, 'VFR')
# tagFlightRules(vfrOverflight, 'VFR Overflight')
# tagLocal()
# tagTotal()
# removeCommas()
# 
# 
# # Specifically dropping first row where data is irrelavant after formatting
# ops = ops.dropna()
# 
# # Converting Date column to datetime object
# ops['Date'] = pd.to_datetime(ops['Date']).dt.date
# 
# =============================================================================

# =============================================================================
# # Selecting all numeric columns to change type to int64
# numeric_columns = ops.columns[1:]
# # Converting numeric columns from object to int64
# ops[numeric_columns] = ops[numeric_columns].astype('int64')
# 
# # Creating variable for total IFR traffic
# ops['IFR'] = ops['IFR Total'] + ops['IFR Overflight Total']
# 
# # Creating variable for total VFR traffic, which is target variable
# ops['VFR'] = ops['VFR Total'] + ops['VFR Overflight Total'] + ops['Local Total']
# =============================================================================

# # Selecting all numeric columns to change type to int64
numeric_columns2 = dataFrameLst[0][0].columns[1:]
removeCommas2()

dropNA()
# Coverting numeric columns of all ops datasets from object to int64
to_int()
# Creating variable for total IFR and VFR (TARGET VARIABLE) traffic
vfr_ifr()
PRCP_SQRT()
to_datetime()
merged_Lst = merge_datasets()
tagHolidaysLst()
addLocIDLst()
PRCP_SQRT()
#delCol('ELEVATION')
setCols()


# merge ops and weather
# =============================================================================
# ops_weather = ops.merge(weather,on='Date',how='inner') 
# 
# dict1 = {'FAIRBANKS INTERNATIONAL AIRPORT':{'LOC':'FAI','LAT':64.815,'LONG':-147.8563888889},
#          'JUNEAU INTERNATIONAL AIRPORT':{'LOC':'JNU','LAT': 58.355, 'LONG':-134.57639},
#          'ANCHORAGE':{'LOC':'ANC','LAT':61.1741666667, 'LONG':-149.9816666667}
#          }
# =============================================================================

# getting the text of the NAME column
# =============================================================================
# ops_weather.loc[[1]]['NAME'].values[0] #OR
# ops_weather['NAME'].head(1).values[0] # will give 'FAIRBANKS INTERNATIONAL AIRPORT, AK US'
# addLocID()
# ops_weather['PRCP_SQRT'] = ops_weather['PRCP']**(1./2)
# tagHolidays()       
# ops_weather = ops_weather.reindex(columns=column_names)
# #data_dict = {'ANC':ops_weather} 
# 
# =============================================================================

def trimDatasets():
    datasets = create_dataDict()
    iterator = iter(datasets)
    nxt = next(iterator)
    
    cols1 = datasets[nxt].columns[:6]
    cols2 = datasets[nxt].columns[-9:]

    
    x4 = [col for col in cols1]
    x5 = [col for col in cols2]
    x6 = x4+x5
    
    for key in datasets:
        datasets[key] = datasets[key].reindex(columns=x6)
    
    return datasets

# The following variable can be used to access any cleaned dataset by referencing the airport's LOC
# For example: datasets['FAI'].head() will return
#
# =============================================================================
#          Date  LOC      STATION                                    NAME  \
# 0  2017-01-01  FAI  USW00026411  FAIRBANKS INTERNATIONAL AIRPORT, AK US   
# 1  2017-01-02  FAI  USW00026411  FAIRBANKS INTERNATIONAL AIRPORT, AK US   
# 2  2017-01-03  FAI  USW00026411  FAIRBANKS INTERNATIONAL AIRPORT, AK US   
# 3  2017-01-04  FAI  USW00026411  FAIRBANKS INTERNATIONAL AIRPORT, AK US   
# 4  2017-01-05  FAI  USW00026411  FAIRBANKS INTERNATIONAL AIRPORT, AK US   
# 
#    LATITUDE  LONGITUDE  VFR  IFR  AWND  PRCP  PRCP_SQRT  SNOW  TMIN  TMAX  \
# 0  64.80309 -147.87606   38   44  1.12  0.00   0.000000   0.0    -5    20   
# 1  64.80309 -147.87606   93   95  2.24  0.00   0.000000   0.0    -6     8   
# 2  64.80309 -147.87606   47  116  2.46  0.06   0.244949   1.0    -7    17   
# 3  64.80309 -147.87606   36  121  0.89  0.08   0.282843   1.0    15    19   
# 4  64.80309 -147.87606   28  107  9.84  0.16   0.400000   3.7    -9    24   
# 
#    isAHoliday  
# 0         1.0  
# 1         0.0  
# 2         0.0  
# 3         0.0  
# 4         0.0  
# =============================================================================

datasets = trimDatasets()

cols_w_blanks = []

def findWeatherBlanks():
    global cols_w_blanks
    target_key = ''
    blanks = {}
    for key in datasets:
        dset = datasets[key]
        dset_counts = dset.describe().iloc[[0]]
        for col in dset_counts.columns:
            # if the column's count attribute is less than 1826 (the length of the dataset)
            # that column has blank values that needs to be addressed
            if dset_counts.loc['count',col] < len(dset):
                target_key = key
                #print('{}: {} -- {}'.format(key,col,dset_counts.loc['count',col]))
                #cols_w_blanks += [dset[[col]]]
                cols_w_blanks += [col]
        if target_key != '':
            blanks[target_key] = [cols_w_blanks[i] for i in range(len(cols_w_blanks))]

    return blanks


def meanImputer():
    for key in datasets:
        if key in blanks:
            lst = blanks[key]
            for col in lst:
                avg_val = round(np.mean(datasets[key][col]),2)
                datasets[key][col] = datasets[key][col].fillna(avg_val)

blanks = findWeatherBlanks()

# returs final dictionary of dataframes after nan values are imputed with column-wise
# mean values
meanImputer()
