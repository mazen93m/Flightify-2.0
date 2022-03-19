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
openweather_api = 'https://api.openweathermap.org/data/2.5/onecall?exclude=hourly,alerts,minutely&appid={api key}&units=imperial'


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
    

#Basic Ending 1.0  
#No special treatment.If a variable isn't there such as prcp or 
#snow the system stops and throws an error.    
    
#Now that the test works create lists for the forcast variables
#list names need to be lower case
    
date, tmax, awnd, forecast, prob_prec = [], [], [], [], []
#prcp, snow = [], []

for day in (response['daily']):
    date.append(day['dt'])
    tmax.append(day['temp']['max'])
    awnd.append(day['wind_speed'])
    forecast.append(day['weather'][0]['main'])
    prob_prec.append(day['pop'])
#   prcp.append(day['rain'])
#   snow.append(day['snow'])
    
#Make a pandas dataframe from the list created from the json
#response 
    
forecast_df = pd.DataFrame(
    {
        'DATE_unix_UTC': date,
        'TMAX_F': tmax,
        'AWND_mph': awnd,
        'forecast': forecast,
        'prob_prec': prob_prec
      # 'PRCP_mm': prcp,
      # 'SNOW_mm': snow
    }   
)
forecast_df



#--------------------------

#Ending 2.0
#Checking if KeyError.
#If KeyError it places a 0 for rain and 0 for snow
# and NaN for others


#Now that the test works create lists for the forcast variables
#list names need to be lower case
    
date, tmax, awnd, forecast, prob_prec = [], [], [], [], []
prcp, snow = [], []

for day in (response['daily']):
    
    try:
        date.append(day['dt'])
    except KeyError:
        print('... Open Weather API unresponsive for date, try again.')
        date.append('NaN')
        
    try:    
        tmax.append(day['temp']['max'])
    except KeyError:
        print('... Open Weather API unresponsive for tmax, try again.')
        tmax.append('NaN')
        
    try:
        awnd.append(day['wind_speed'])
    except KeyError:
        print('... Open Weather API unresponsive for awnd, try again.')
        awnd.append('NaN')
        
    try:    
        forecast.append(day['weather'][0]['main'])
    except KeyError:
        print('... Open Weather API unresponsive for forecast, try again.')
        forecast.append('NaN') 
        
    try:    
        prob_prec.append(day['pop'])
    except KeyError:
        print('... Open Weather API unresponsive for prob_prec, try again.')
        prob_prec.append('NaN')  
        
    try:
        prcp.append(day['rain'])
    except KeyError:
        print('... Open Weather API unresponsive for prcp, try again.')
        prcp.append(0)
        
    try:    
        snow.append(day['snow'])
    except KeyError:
        print('... Open Weather API unresponsive for snow, try again.')
        snow.append('0')   
    
#Make a pandas dataframe from the list created from the json
#response 
    
forecast_df = pd.DataFrame(
    {
        'DATE_unix_UTC': date,
        'TMAX_F': tmax,
        'AWND_mph': awnd,
        'forecast': forecast,
        'prob_prec': prob_prec,
        'PRCP_mm': prcp,
        'SNOW_mm': snow
    }   
)
forecast_df


#--------------------------

#Ending 3.0
#Checking if KeyError and create binary prcp and binary snow.
#If KeyError place NaN for all except prcp and snow.
#For prcp and snow make binary depending on forecast result.



#Create a rain and snow code list to use in binary prcp and snow
#variable creation
rain_list = ['Thunderstorm', 'Drizzle', 'Rain', ' Mist', 'Fog']
snow_list = ['Snow']
    

#Now that the test works create lists for the forcast variables
#list names need to be lower case


date, tmax, awnd, forecast, prob_prec = [], [], [], [], []
prcp, snow = [], []

for day in (response['daily']):
    
    try:
        date.append(day['dt'])
    except KeyError:
        print('... Open Weather API unresponsive for date, try again.')
        date.append('NaN')
        
    try:    
        tmax.append(day['temp']['max'])
    except KeyError:
        print('... Open Weather API unresponsive for tmax, try again.')
        tmax.append('NaN')
        
    try:
        awnd.append(day['wind_speed'])
    except KeyError:
        print('... Open Weather API unresponsive for awnd, try again.')
        awnd.append('NaN')
        
    try:    
        forecast.append(day['weather'][0]['main'])
    except KeyError:
        print('... Open Weather API unresponsive for forecast, try again.')
        forecast.append('NaN') 
        
    try:    
        prob_prec.append(day['pop'])
    except KeyError:
        print('... Open Weather API unresponsive for prob_prec, try again.')
        prob_prec.append('NaN')  
        
    try:
        prcp.append(int(day['weather'][0]['main'] in rain_list))
    except KeyError:
        print('... Open Weather API unresponsive for prcp, try again.')
        prcp.append(0)
        
    try:    
        snow.append(int(day['weather'][0]['main'] in snow_list))
    except KeyError:
        print('... Open Weather API unresponsive for snow, try again.')
        snow.append('0')   
    
#Make a pandas dataframe from the list created from the json
#response 
    
forecast_df = pd.DataFrame(
    {
        'DATE_unix_UTC': date,
        'TMAX_F': tmax,
        'AWND_mph': awnd,
        'forecast': forecast,
        'prob_prec': prob_prec,
        'PRCP_binary': prcp,
        'SNOW_binary': snow
    }   
)
forecast_df






