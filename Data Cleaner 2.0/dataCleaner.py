#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:46:44 2022

@author: ajgray
"""

import pandas as pd
import numpy as np
import airports as a
import os
import pickle


os.chdir('/Users/ajgray/Desktop/project')

# Setting any output to display all columns and rows 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

airports = a.airports
dataFrameLst = []

#airports = {'BCT': ['bdl_opsnet_tower_ops_2017-2021.csv','bdl_noaa_usw00014740_2017-2021.csv','HARTFORD BRADLEY INTERNATIONAL AIRPORT, CT US']}
#airports = {'ANC': ['anc_opsnet_tower_ops_2017-2021.csv','anc_noaa_usw00026451_2017-2021.csv','anchorage ted stevens international airport, ak us']}

duplicate_airports = a.duplicate_airports

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
        dataFrameLst.append([read_csv('/Users/ajgray/Desktop/project/tower_ops_airports/{}'.format(airports[nxt][0]),6,0),read_csv('/Users/ajgray/Desktop/project/NOAA/{}'.format(airports[nxt][1])), [airports[nxt]]])
        if nxt in duplicate_airports:
            dataFrameLst[i][1].NAME = duplicate_airports[nxt]
        if i == len(airports):
            break
    
    return dataFrameLst

create_dataFrames()
holidays = read_csv('updated_holidays.csv')


def renameDate(data):
    for el in data:
        el[1] = el[1].rename(columns={'DATE':'Date'})

def to_datetime(data):
    for el in data:
        # ops data Date column
        el[0]['Date'] = pd.to_datetime(el[0]['Date']).dt.date
        # weather data Date column
        el[1]['Date'] = pd.to_datetime(el[1]['Date']).dt.date

    holidays['Date'] = pd.to_datetime(holidays['Date']).dt.date

def dropLastCol(data):
    for el in data:
        if len(el[0].columns) == 27:
            el[0].drop(el[0].columns[-1], axis=1, inplace=True)

def dropLast5Rows(data):
    for el in data:
        el[0] = el[0].iloc[:-5,:]
    
renameDate(dataFrameLst)
dropLastCol(dataFrameLst)
dropLast5Rows(dataFrameLst)

def ifrItinerant_Lst(data):
    ifrCleaned = tagFlightRulesLst(ifrItinerant2, 'IFR')
    first_El = [[df[0]] for df in data]
    second_El = [[df[1]] for df in data]

    for el in first_El:
        for i in range(5):
            el[0] = el[0].rename(columns={el[0].columns[1:6][i]:ifrCleaned[i]})
    
    dataFrameLst2 = []
    for i in range(len(first_El)):
        dataFrameLst2+=[[first_El[i][0], second_El[i][0]]]
        
    return dataFrameLst2

def ifrOverflight_Lst(dfLst):
    ifrOvCleaned = tagFlightRulesLst(ifrOverflight2, 'IFR Overflight')
    first_El = [[df[0]] for df in dfLst]
    second_El = [[df[1]] for df in dfLst]

    for el in first_El:
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
            el[0] = el[0].rename(columns={el[0].columns[24:][i]:totalCleaned[i]})
        
    dataFrameLst2 = []
    for i in range(len(first_El)):
        dataFrameLst2+=[[first_El[i][0], second_El[i][0]]]
    
    return dataFrameLst2

def tagColumns(data):
    
    # tagging IFR for the itinerant columns
    dfLst_ifr = ifrItinerant_Lst(data)
    dfLst_ifr_ovfl = ifrOverflight_Lst(dfLst_ifr)
    dfLst_ifr_ovfl_vfr = vfr_Lst(dfLst_ifr_ovfl)
    dfLst_ifr_ovfl_vfr_ovfl = vfrOverflight_Lst(dfLst_ifr_ovfl_vfr)
    dfLst_ifr_ovfl_vfr_ovfl_local = local_Lst(dfLst_ifr_ovfl_vfr_ovfl)
    
    dfLst = total_Lst(dfLst_ifr_ovfl_vfr_ovfl_local)
    
    return dfLst

def delCol(*cols):
    for col in cols:
        for dataset in merged_Lst:
            if col in [col for dataset in merged_Lst for col in dataset.columns]:
                del dataset[col]
    
# Parse out and group individual columns for relabelling
ifrItinerant2 = dataFrameLst[0][0].columns[1:6]
ifrOverflight2 = dataFrameLst[0][0].columns[6:11]

vfrItinerant2 = dataFrameLst[0][0].columns[11:16]
vfrOverflight2 = dataFrameLst[0][0].columns[16:21]

local2 = dataFrameLst[0][0].columns[21:24]
total2 = dataFrameLst[0][0].columns[24:]


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

def tagTotal2():
    '''
        Tag Total variables
    '''
    total2.values[-2] = 'Total Airport Operations'
    total2.values[-1] = 'Total Tower Operations'
    
    return total2

dfLst = tagColumns(dataFrameLst)

def removeCommas2(data):
    '''
    This function accesses the row of each non-Date column of the DataFrame, checks to see if
    that row's length is greater than 3 (i.e. 1,001), and modifies that row
    by removing the comma.

    '''
    
    for el in data:
        for col in el[0].columns:
            if col.lower() != 'date' and el[0][col].dtype == 'O':
                for i in range(1,len(el[0][col])):
                    if len(el[0][col].loc[i]) > 3:
                        el[0].loc[i, col] = el[0][col].loc[i].replace(',','')

                
def dropNA(data):
    for el in data:
        el[0] = el[0].dropna()


def addLocIDLst(base, lst):
    '''
    This function uses a dictionary to match the airport to its respective airport code.
    It then adds a column to the merged dataset equal to the airport code.
    
    '''
    
    for key in base.keys():
        for dataset in lst:
            #print(dataset.loc[0]['NAME'])
            if base[key][2].upper() == dataset.loc[0]['NAME']:
                dataset['LOC'] = key.upper()

def tagHolidaysLst(lst):
    '''
    This function loops through the ops_weather merged dates and tries to find matches in the
    holidays dates. 
    If there is a match, the ops_weather dataset is updated:
        1. A new coloumn (isAHoliday) is assigned the value 1, else zero. 
        2. A new column (Holiday) is assinged the appropriate holiday for that date

    '''
    
    holDates = list(holidays['Date'])
    
    for el in lst:
        for target_date in range(len(el)):
            if el.loc[target_date, 'Date'] not in holDates:
                el.loc[target_date, 'isAHoliday'] = 0
            else:
                el.loc[target_date,'isAHoliday'] = 1

def to_int(data):
    for el in data:
        el[0][numeric_columns2] = el[0][numeric_columns2].astype('int64')


def vfr_ifr(data):
    for el in data:
        el[0]['VFR'] = el[0]['VFR Total'] + el[0]['VFR Overflight Total']
        el[0]['IFR'] = el[0]['IFR Total'] + el[0]['IFR Overflight Total']
        
def PRCP_SQRT(data):
    for el in data:
        el[1]['PRCP_SQRT'] = el[1]['PRCP']**(1./2)
        
    
# =============================================================================
# def PROB_PRCP():
#     for el in dfLst:
#         for i in range(len(el)):   
#             if (el[1].loc[i, 'PRCP'] > 0) or ('SNOW' in el[1].columns.values and el[1].loc[i, 'SNOW'] > 0):   
#                 el[1].loc[i, 'PROB_PRCP'] = 1
#             else:
#                 el[1].loc[i, 'PROB_PRCP'] = 0
# =============================================================================
merged_Lst = []

def merge_datasets(data, lst):
    for el in data:
        merged_dataset = el[0].merge(el[1],on='Date',how='inner')
        lst+=[merged_dataset]
        
    return lst

def setCols(lst):
    # set merged dataset columns an ordered subset list of columns
    for i in range(len(lst)):
        lst[i] = lst[i].reindex(columns=column_names)

# save data to a dictionary
def create_dataDict(lst):
    
    keys = [lst[i]['LOC'][0] for i in range(len(lst))]
    values = [lst[i] for i in range(len(lst))]
    datasets = {}
    
    for i in range(len(lst)):
        datasets[keys[i]] = values[i]
    
    return datasets
        


# # Selecting all numeric columns to change type to int64
numeric_columns2 = dataFrameLst[0][0].columns[1:]
removeCommas2(dfLst)

dropNA(dfLst)
# Coverting numeric columns of all ops datasets from object to int64
to_int(dfLst)
# Creating variable for total IFR and VFR (TARGET VARIABLE) traffic
vfr_ifr(dfLst)
PRCP_SQRT(dfLst)
to_datetime(dfLst)
#PROB_PRCP()
merge_datasets(dfLst, merged_Lst)
tagHolidaysLst(merged_Lst)
addLocIDLst(airports, merged_Lst)
#delCol('ELEVATION')
setCols(merged_Lst)


def trimDatasets(lst):
    datasets = create_dataDict(lst)
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

datasets = trimDatasets(merged_Lst)
cols_w_blanks = []

def findWeatherBlanks(data, lst):
    target_key = ''
    blanks = {}
    for key in data:
        dset = data[key]
        dset_counts = dset.describe().iloc[[0]]
        for col in dset_counts.columns:
            # if the column's count attribute is less than 1826 (the length of the dataset)
            # that column has blank values that needs to be addressed
            if dset_counts.loc['count',col] < len(dset):
                target_key = key

                lst += [col]
        if target_key != '':
            blanks[target_key] = [lst[i] for i in range(len(lst))]

    return blanks


def meanImputer(data):
    for key in data:
        if key in blanks:
            lst = blanks[key]
            for col in lst:
                avg_val = round(np.mean(data[key][col]),2)
                data[key][col] = data[key][col].fillna(avg_val)
                


blanks = findWeatherBlanks(datasets, cols_w_blanks)

# returs final dictionary of dataframes after nan values are imputed with column-wise
# mean values
meanImputer(datasets)



def findNaNCols():
    dropped_dict = {}
    for key in datasets:
        dropped_lst = []
        for col in datasets[key].describe().loc[['count']]:
            if datasets[key].describe().loc['count',col] == 0:
               dropped_lst.append(col)
        
        if len(dropped_lst) > 0:
            dropped_dict[key] = dropped_lst
        
            
    
    return dropped_dict

dropped_dict = findNaNCols()

def drop_NA_Cols():
    for key in dropped_dict:
        datasets[key] = datasets[key].drop(columns=[col for col in dropped_dict[key]],axis=1)

def zeroImputer():
    for key in dropped_dict:
        for col in dropped_dict[key]:
            datasets[key][col] = datasets[key][col].fillna(0)

def SNOW_SQRT():
    for df in datasets:
        if 'SNOW' in datasets[df].columns:
            datasets[df]['SNOW_SQRT'] = datasets[df]['SNOW']**(1./2)
            
def dropHoliday():
    for key in datasets:
        datasets[key] = datasets[key].drop(['isAHoliday'], axis=1)
          
def parseDateCols():
    for key in datasets:
        data = datasets[key]                
        # Create day, month, and weekday predictor variables
        for i in range(len(data)):
            data.loc[i, 'Year'] = data.loc[i, 'Date'].year
            data.loc[i, 'Month'] = data.loc[i, 'Date'].month
            data.loc[i, 'Day'] = data.loc[i, 'Date'].day
            data.loc[i, 'Week_Day'] = data.loc[i, 'Date'].weekday()
        
        data.index = data.Date
         
        
#drop_NA_Cols()
zeroImputer()
dropHoliday()
parseDateCols()
SNOW_SQRT()



# load pickle module

# create a binary pickle file
f = open("datasets.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(datasets,f)

# close file
f.close()
