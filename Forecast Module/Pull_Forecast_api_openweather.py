# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:23:36 2022

@author: Kimberly Cawi
"""
#code inspired from
#https://python.plainenglish.io/how-to-get-weather-forecast-data-in-your-jupyter-notebook-d93a092d63ec
#How to Get Weather Forecast Data in Jupyter Notebook

#This script pulls forecast weather data from the Open Weather API,
#receives a json response and places specific values in a pandas
#dataframe.

#Main Open Weather API site
#https://openweathermap.org/

#This script provides daily forecast values for the current
# day plus the next seven days.

#You will need an Open Weather Key Code.  There are many key options
#with different benefits.

#I suggest to begin with the free key code.
#Subscribe here: https://openweathermap.org/api
#Click Subscribe under One Call API
#(API doc shows excellent explanation of all fields in API response)
#Choose "Get API Key" Under Free column for 1000 calls/day and
#30,000 calls/month

#List of weather condition codes
# https://openweathermap.org/weather-conditions

#List of all API parameters with available units
#https://openweathermap.org/weather-data

#The call below will return imperial units for temp and wind_speed:
#temp in Fahrenheit,
#the wind_speed in mph.  
#Daily rain in mm
#Daily snow are in mm.
#Daily Probability of precipitation, pop is 0-1
#Daily data receiving time, dt, is in Unix UTC
#Daily weather main is a forecast using descriptive labels, "rain" "snow" etc


#To learn about Unix UTC 
#https://kb.narrative.io/what-is-unix-time
#https://www.epochconverter.com/



import pandas as pd
import requests
import json

#assign API call before adding lat lon
#Important: Insert your API key in place of {api key}, drop braces, no spaces
#Using units=imperial in api call means temp max is F, wind_speed is mph
#rain and snow are returned as mm no matter what

openweather_api = 'https://api.openweathermap.org/data/2.5/onecall?exclude=hourly,alerts,minutely&appid=e1aba7bcfa12974803adcf2db2260958&units=imperial'


#assign lat lon to a string variable
#example: Fairbancks &lat=64.80309&lon=-147.87606 

#Green Bay WI
loc = '&lat=' + str(44.5133) + '&lon=' + str(-88.0133)
 

#Add lat lon to complete the API call   
openweather_url = openweather_api + loc


#Pull the forecast from Open Weather
response = json.loads(requests.get(openweather_url).text)




#the try catch tests json response values for the first response
#under daily which is listed as 0

try:
    date = response['daily'][0]['dt']
    sunrise = response['daily'][0]['sunrise']
    sunset = response['daily'][0]['sunset']
    tmax = response['daily'][0]['temp']['max']
    awnd = response['daily'][0]['wind_speed']
    forecast = response['daily'][0]['weather'][0]['main']
    prob_prec = response['daily'][0]['pop']
    #prcp = response['daily'][0]['rain']
    #snow = response['daily'][0]['snow']    
    print(' ... done.')
except KeyError:
    print('... Open Weather API unresponsive for this request, try again.')
    

# It is possible to have more than one weather condition for
# a requested location. The first weather condition in API 
# response is primary.
# We are using primary  response['daily']['weather'][0]['main']
    




# =============================================================================
# #Basic Ending 1.0  
#     
# #No special treatment.If a variable isn't there such as prcp or 
# #snow the system stops and throws an error.    
#     
# #Now that the test works create lists for the forcast variables
# #list names need to be lower case
#     
# date, tmax, awnd, forecast, prob_prec = [], [], [], [], []
# sunrise, sunset = [], []
# prcp, snow = [], []
# 
# for day in (response['daily']):
#     date.append(day['dt'])
#     sunrise.append(day['sunrise'])
#     sunset.append(day['sunset'])
#     tmax.append(day['temp']['max'])
#     awnd.append(day['wind_speed'])
#     forecast.append(day['weather'][0]['main'])
#     prob_prec.append(day['pop'])
#     #prcp.append(day['rain'])
#     #snow.append(day['snow'])
#     
# 
#    
# #Make a pandas dataframe from the list created from the json
# #response 
#     
# forecast_df = pd.DataFrame(
#     {
#         'DATE_unix_UTC': date,
#         'SUNRISE': sunrise,
#         'SUNSET': sunset,
#         'TMAX': tmax, #in Fahrenheit when api call has units=imperial
#         'AWND': awnd, #in mph when api call has units=imperial
#         'FORECAST': forecast,
#         'PROB_PREC': prob_prec
#       # 'PRCP': prcp, #always in mm
#       # 'SNOW': snow  #always in mm
#     }   
# )
# forecast_df
# 
# 
# =============================================================================

#--------------------------

#Ending 2.0
#Checking if KeyError.
#If KeyError it places a 0 for rain and 0 for snow
# and NaN for others


#Now that the test works create lists for the forcast variables
#list names need to be lower case
    
date, tmax, awnd, forecast, prob_prec = [], [], [], [], []
prcp, snow, sunrise, sunset = [], [], [], []

for day in (response['daily']):
    
    try:
        date.append(day['dt'])
    except KeyError:
        print('... Open Weather API unresponsive for date, inserting NaN.')
        date.append('NaN')
        
    try:
        sunrise.append(day['sunrise'])
    except KeyError:
        print('... Open Weather API unresponsive for sunrise, try again.')
        date.append('NaN')
        
    try:
        sunset.append(day['sunset'])
    except KeyError:
        print('... Open Weather API unresponsive for sunset, try again.')
        date.append('NaN')
        
    try:    
        tmax.append(day['temp']['max'])
    except KeyError:
        print('... Open Weather API unresponsive for temp max, inserting NaN.')
        tmax.append('NaN')
        
    try:
        awnd.append(day['wind_speed'])
    except KeyError:
        print('... Open Weather API unresponsive for wind_speed, inserting NaN.')
        awnd.append('NaN')
        
    try:    
        forecast.append(day['weather'][0]['main'])
    except KeyError:
        print('... Open Weather API unresponsive for weather main, inserting NaN.')
        forecast.append('NaN') 
        
    try:    
        prob_prec.append(day['pop'])
    except KeyError:
        print('... Open Weather API unresponsive for pop, inserting NaN.')
        prob_prec.append('NaN')  
        
    try:
        prcp.append(day['rain'])
    except KeyError:
        print('... Open Weather API unresponsive for rain, inserting 0.')
        prcp.append(0)
        
    try:    
        snow.append(day['snow'])
    except KeyError:
        print('... Open Weather API unresponsive for snow, inserting 0.')
        snow.append(0)   
    
#Make a pandas dataframe from the list created from the json
#response 
    
forecast_df = pd.DataFrame(
    {
        'DATE_unix_UTC': date,
        'SUNRISE': sunrise,
        'SUNSET': sunset,
        'TMAX': tmax, #in Fahrenheit when api call has units=imperial
        'AWND': awnd, #in mph when api call has units=imperial
        'FORECAST': forecast,
        'PROB_PREC': prob_prec,
        'PRCP': prcp, #always in mm
        'SNOW': snow  #always in mm
    }   
)
forecast_df






# =============================================================================
# #Ending 3.0
# #Checking if KeyError and create binary prcp and binary snow.
# #If KeyError place NaN for all except prcp and snow.
# #For prcp and snow make binary depending on forecast result.
# 
# 
# 
# #Create a rain and snow code list to use in binary prcp and snow
# #variable creation
# 
# rain_list = ['Thunderstorm', 'Drizzle', 'Rain', ' Mist', 'Fog']
# snow_list = ['Snow']
#     
# 
# #Now that the test works create lists for the forcast variables
# #list names need to be lower case
# 
# 
# date, tmax, awnd, forecast, prob_prec = [], [], [], [], []
# prcp, snow, sunrise, sunset = [], [], [], []
# 
# for day in (response['daily']):
#     
#     try:
#         date.append(day['dt'])
#     except KeyError:
#         print('... Open Weather API unresponsive for dt, try again.')
#         date.append('NaN')
#      
#     try:
#         sunrise.append(day['sunrise'])
#     except KeyError:
#         print('... Open Weather API unresponsive for sunrise, try again.')
#         date.append('NaN')
#         
#     try:
#         sunset.append(day['sunset'])
#     except KeyError:
#         print('... Open Weather API unresponsive for sunset, try again.')
#         date.append('NaN')
#                
#     try:    
#         tmax.append(day['temp']['max'])
#     except KeyError:
#         print('... Open Weather API unresponsive for temp max, try again.')
#         tmax.append('NaN')
#         
#     try:
#         awnd.append(day['wind_speed'])
#     except KeyError:
#         print('... Open Weather API unresponsive for wind_speed, try again.')
#         awnd.append('NaN')
#         
#     try:    
#         forecast.append(day['weather'][0]['main'])
#     except KeyError:
#         print('... Open Weather API unresponsive for weather main, try again.')
#         forecast.append('NaN') 
#         
#     try:    
#         prob_prec.append(day['pop'])
#     except KeyError:
#         print('... Open Weather API unresponsive for pop, try again.')
#         prob_prec.append('NaN')  
#         
#     try:
#         prcp.append(int(day['weather'][0]['main'] in rain_list))
#     except KeyError:
#         print('... Open Weather API unresponsive for prcp binary calc, try again.')
#         prcp.append('NaN')
#         
#     try:    
#         snow.append(int(day['weather'][0]['main'] in snow_list))
#     except KeyError:
#         print('... Open Weather API unresponsive for snow binary calc, try again.')
#         snow.append('NaN')   
#     
# #Make a pandas dataframe from the list created from the json
# #response 
#     
# forecast_df = pd.DataFrame(
#     {
#         'DATE_unix_UTC': date,
#         'SUNRISE': sunrise,
#         'SUNSET': sunset,
#         'TMAX': tmax, #in Fahrenheit when api call has units=imperial
#         'AWND': awnd, #in mph when api call has units=imperial
#         'FORECAST': forecast,
#         'PROB_PREC': prob_prec,
#         'PRCP': prcp, #always in mm
#         'SNOW': snow  #always in mm
#     }   
# )
# forecast_df
# 
# =============================================================================
#---------------------------

#Convert unix date time to local date time
#It is converted to the local date and time by default, 

#Code from https://note.nkmk.me/en/python-unix-time-datetime/

#"Use datetime.fromtimestamp() of the datetime module
# to convert Unix time to datetime object. Specify Unix time as
# an argument."


import datetime

#check how it works
#dt = datetime.datetime.fromtimestamp(0)

#print(dt)
# 1970-01-01 09:00:00

#print(type(dt))
# <class 'datetime.datetime'>

#print(dt.tzinfo)
# None




# =============================================================================
# #Convert our forecast unix dates to local date time using datetime package
# #result is datetime.datetime type
# 
# print(forecast_df.loc[:,"DATE_unix_UTC"])
# 
# Date = []  #place to hold converts to local time
# 
# for i in forecast_df.loc[:,'DATE_unix_UTC']:
#     convert = datetime.datetime.fromtimestamp(i) #default is local time zone
#     Date.append(convert)
#     
# print(Date) 
# 
# type(Date[0])
# 
# forecast_df['Date'] = Date
# =============================================================================





#Convert our forecast unix dates to local date time using pandas
#result is class 'pandas._libs.tslibs.timestamps.Timestamp'

Date = []  #place to hold converts to local time

for i in forecast_df.loc[:,'DATE_unix_UTC']:
    convert = pd.to_datetime(i, unit='s') #default is local time zone
    Date.append(convert)
    
print(Date) 

print(type(Date[0]))

forecast_df['Date'] = Date





#Convert sunrise unix date time to Local date time

#print(forecast_df.loc[:,'SUNRISE']) #in unix

SUNRISE_local = []  #place to hold converts to local time

for i in forecast_df.loc[:,'SUNRISE']:
    convert = datetime.datetime.fromtimestamp(i) #default is local time zone
    SUNRISE_local.append(convert)
 
type(SUNRISE_local[0])
    
#print(SUNRISE_local)  

forecast_df['SUNRISE'] = SUNRISE_local  #replace unix SUNRISE with local converts

type(forecast_df['SUNRISE'][0])





#Convert sunset unix date time to Local date time

#print(forecast_df.loc[:,'SUNSET']) #in unix

SUNSET_local = []  #place to hold converts to local time

for i in forecast_df.loc[:,'SUNSET']:
    convert = datetime.datetime.fromtimestamp(i) #default is local time zone
    SUNSET_local.append(convert)
    
#print(SUNSET_local)  

forecast_df['SUNSET'] = SUNSET_local  #replace unix SUNRISE with local converts




#--------------------------------------------------

#Convert PRCP from mm to inches and replace


#print(forecast_df.loc[:,'PRCP']) #in mm

PRCP_inches = []  #place to hold converts to inches

for i in forecast_df.loc[:,'PRCP']:
    convert = i/25.4 #
    PRCP_inches.append(convert)
    
#print(PRCP_inches)  

forecast_df['PRCP'] = PRCP_inches  #replace  PRCP in mm with PRCP in inches


#Create a PRCP_SQRT

forecast_df['PRCP_SQRT'] = forecast_df['PRCP'] ** (1/2)


#--------------------------------------------------


#Convert SNOW from mm to inches and replace

#print(forecast_df.loc[:,'SNOW']) #in mm

SNOW_inches = []  #place to hold converts to inches

for i in forecast_df.loc[:,'SNOW']:
    convert = i/25.4 #
    SNOW_inches.append(convert)
    
#print(SNOW_inches)  

forecast_df['SNOW'] = SNOW_inches  #replace  SNOW in mm with SNOW in inches


#Create a SNOW_SQRT

forecast_df['SNOW_SQRT'] = forecast_df['SNOW'] ** (1/2)




#--------------------------------------------------





#Create a Holiday Column

#read in the holiday csv with dates before and after the holiday and weekend

holidays = pd.read_csv('near_weekend_holiday_dates.csv')

#change to date with pandas
#result is class 'pandas._libs.tslibs.timestamps.Timestamp'
holidays['Date'] = pd.to_datetime(holidays['Date'])

#check type
type(holidays['Date'][0])

  



#----------------------------------





#To recognize that two dates are the same, we need to get rid of the time
#in the date

#Convert class 'pandas._libs.tslibs.timestamps.Timestamp' to just date part
#for forecast dates

#test it
pd.Timestamp.date(forecast_df['Date'][0])
pd.Timestamp.date(holidays['Date'][0])   

#convert it
Date = []

for i in forecast_df.loc[:,'Date']:
    Date.append(pd.Timestamp.date(i))

type(forecast_df['Date'][0])

forecast_df['Date'] = Date  

type(forecast_df['Date'][0])


#Convert class 'pandas._libs.tslibs.timestamps.Timestamp' to just date part
#for holidays dataframe dates


Date = []

for i in holidays.loc[:,'Date']:
    Date.append(pd.Timestamp.date(i))

type(holidays['Date'][0])

holidays['Date'] = Date  

type(holidays['Date'][0])



#holidays['Date']
#forecast_df['Date']




isAHOLIDAY = [] 

for i in range(len(forecast_df['Date'])):
    tag = 0
    for j in range(len(holidays['Date'])):
         if holidays['Date'][j] == forecast_df['Date'][i]:
             tag = tag + 1
    if tag > 0:
        isAHOLIDAY.append(1)
    else:
        isAHOLIDAY.append(0)         

isAHOLIDAY

forecast_df['isAHOLIDAY'] = isAHOLIDAY

