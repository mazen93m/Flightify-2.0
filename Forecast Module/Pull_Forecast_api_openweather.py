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
import pickle
import os

# =============================================================================
# #Read in model dictionary from disk
# 

file_to_read = open("model_dict.pkl", "rb")
# 
model_dict = pickle.load(file_to_read)

file_to_read.close()
# 
# =============================================================================



#assign API call before adding lat lon
#Important: Insert your API key in place of {api key}, drop braces, no spaces
#Using units=imperial in api call means temp max is F, wind_speed is mph
#rain and snow are returned as mm no matter what

openweather_api = 'https://api.openweathermap.org/data/2.5/onecall?exclude=hourly,alerts,minutely&appid=e1aba7bcfa12974803adcf2db2260958&units=imperial'

#start a forecast dictionary
forecast_dict = {}

#start a combined forcast dataaframe to hold all forecast dataframes to
combo_forecast_df = pd.DataFrame()
#combo_forecast_df


#Iterate over model_dict or choose an airport to build forecast_dict

#for key in ['FAI']:
for key in model_dict:
    
    print("Running airport " + key)
   
    #assign lat lon to a string variable
    #example: Fairbancks &lat=64.80309&lon=-147.87606 

    lat = str(model_dict[key][1])
    lon = str(model_dict[key][2])
    
    #Green Bay WI
    #loc = '&lat=' + str(44.5133) + '&lon=' + str(-88.0133)
     
    loc = '&lat=' + lat + '&lon=' + lon
    
    #Add lat lon to complete the API call   
    openweather_url = openweather_api + loc
    
    
    #Pull the forecast from Open Weather
    response = json.loads(requests.get(openweather_url).text)
    
    
    
    #test api
    #the try catch tests json response values for the first response
    #under daily which is listed as 0
    
    try:
        date = response['daily'][0]['dt']
        sunrise = response['daily'][0]['sunrise']
        sunset = response['daily'][0]['sunset']
        tmax = response['daily'][0]['temp']['max']
        tmin = response['daily'][0]['temp']['min']
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
        
    
    
    #--------------------------
    
    #Ending 1.0
    #Checking if KeyError.
    #If KeyError it places a 0 for rain and 0 for snow
    # and NaN for others
    
    
    #Now that the test works create lists for the forcast variables
    #list names need to be lower case
    
    # Use rain_List and snow_List if choosing binary prcp and snow
    # rain_list = ['Thunderstorm', 'Drizzle', 'Rain', ' Mist', 'Fog']
    # snow_list = ['Snow']
        
    date, tmax, tmin, awnd, forecast, prob_prec = [], [], [], [], [], []
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
            tmin.append(day['temp']['min'])
        except KeyError:
            print('... Open Weather API unresponsive for temp min, inserting NaN.')
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
            
    #code below creates a binary prcp and binary snow depending on forecast results
    #   try:
    #       prcp.append(int(day['weather'][0]['main'] in rain_list))
    #   except KeyError:
    #       print('... Open Weather API unresponsive for prcp binary calc, try again.')
    #       prcp.append('NaN')
    #         
    #   try:    
    #       snow.append(int(day['weather'][0]['main'] in snow_list))
    #   except KeyError:
    #       print('... Open Weather API unresponsive for snow binary calc, try again.')
    #       snow.append('NaN')   
                    
        
    #Make a pandas dataframe from the list created from the json
    #response 
        
    forecast_df = pd.DataFrame(
        {
            'DATE_unix_UTC': date,
            'SUNRISE': sunrise,
            'SUNSET': sunset,
            'TMAX': tmax, #in Fahrenheit when api call has units=imperial
            'TMIN': tmin, #in Fahrenheit when api call has units=imperial
            'AWND': awnd, #in mph when api call has units=imperial
            'FORECAST': forecast,
            'PROB_PREC': prob_prec,
            'PRCP': prcp, #always in mm
            'SNOW': snow  #always in mm
        }   
    )
    forecast_df
    
    
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
        
    #print(Date) 
    
    #print(type(Date[0]))
    
    forecast_df['Date'] = Date
    
    
    
    #Create Columns for Month Day and Weekday
    
    Month = list(pd.DatetimeIndex(forecast_df['Date']).month)
    Day = list(pd.DatetimeIndex(forecast_df['Date']).day)  
    Week_Day = list(pd.DatetimeIndex(forecast_df['Date']).weekday)
    
    forecast_df['Month'] = Month
    forecast_df['Day'] = Day
    forecast_df['Week_Day'] = Week_Day
    
    
    
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
    

            
#------------------------------------
    
    #Create IFR column in forecast_df uses average from data cleaner's datasets
    #which was brought in with the model_dict
    
    IFR = []
    
    for i in range(len(forecast_df['Date'])):
        IFR.append(model_dict[key][3])
        
    forecast_df['IFR'] = IFR  
    
#----------------------------------------
        
    #Create Intercept column of 1's in forecast_df
    
    Intercept = []
    
    for i in range(len(forecast_df['Date'])):
        Intercept.append(1)
        
    forecast_df['Intercept'] = Intercept 
    
#----------------------------------------
       
    #Create a LOC column in forecast_df 
    
    LOCid = []
    
    for i in range(len(forecast_df['Date'])):
        LOCid.append(key)
    
    forecast_df['LOC'] = LOCid
    
    #forecast_df.columns
     
#--------------------------------------

    #Create LATITUDE column in forecast_df
    
    LATITUDE = []
    
    for i in range(len(forecast_df['Date'])):
        LATITUDE.append(model_dict[key][1])
    
    forecast_df['LATITUDE'] = LATITUDE
       
#--------------------------------------

    #Create LONGITUDE column in forecast_df
    LONGITUDE = []
    
    for i in range(len(forecast_df['Date'])):
        LONGITUDE.append(model_dict[key][2])
    
    forecast_df['LONGITUDE'] = LONGITUDE
     
#-------------------------------------- 

    possible_predictors = ['IFR', 'AWND', 'PRCP', 'PRCP_SQRT', 'SNOW', 'SNOW_SQRT',
                           'TMIN', 'TMAX', 'isAHOLIDAY', 'Month', 'Day', 'Week_Day']
    
    #create a column place holder for intercept and coefficeints
    forecast_df['Intercept_coeff'] = 'NaN'
    for i in range(len(possible_predictors)):
        forecast_df[possible_predictors[i] + '_coeff'] = 'NaN' 
        
    #create a column place holder for intercept and coefficeints standard errors
    forecast_df['Intercept_coeff_se'] = 'NaN'
    for i in range(len(possible_predictors)):
        forecast_df[possible_predictors[i] + '_coeff_se'] = 'NaN' 
               
    #create a column place holder for intercept and coefficeints p values
    forecast_df['Intercept_p'] = 'NaN'
    for i in range(len(possible_predictors)):
        forecast_df[possible_predictors[i] + '_p'] = 'NaN'
        
    #create a column place holder for test_root_MSE
    forecast_df['test_root_MSE'] = 'NaN' 
    
    #create a column place holder for R2
    forecast_df['R2'] = 'NaN' 
    
    #create a column place holder for PSEUDO R-SQU
    forecast_df['PSEUDO R-SQU'] = 'NaN'
    
    #create a column place holder for LOG-LIKELIHOOD
    forecast_df['LOG-LIKELIHOOD'] = 'NaN'
    
    #create a column place holder for LL-NULL
    forecast_df['LL-NULL'] = 'NaN'
    
    #create a column place holder for LLR p-value
    forecast_df['LLR p-value'] = 'NaN'
    
    #create a confidence lower limit place holder for prediction
    forecast_df['CONF_lower'] = 'NaN'
    
    
    #create a confidence upper limit place holder for prediction
    forecast_df['CONF_upper'] = 'NaN'
    
#--------------------------------------
 
        
        
#--------------------------------------
      
    #Forecast current plus next 7 days of VFR Traffic
         
    #Will need to check which model is used and build x_forecast with appropriate columns
    
    #Forecast Prediction code for Model MLR1
    if model_dict[key][0] == 'MLR1':
        
        #Calculate Future VFR flights
        
        predictor_names = model_dict[key][6]
        #predictor_names = ['IFR', 'AWND', 'PRCP_SQRT', 'TMAX', 'isAHOLIDAY', 'SNOW_SQRT']
        
        #create x value DataFrame for predictions
        x_forecast = forecast_df[predictor_names]
    
        #x_forecast = x_forecast.to_numpy()
        
        #Use model object from model_dict to forecast current plus next 7 days
        linReg = model_dict[key][4]
        y_forecast = linReg.predict(x_forecast)
        
        forecast_df['y_forecast'] = y_forecast
        
        
        #populate intercept and coefficients columns in forecast_df with model values
        forecast_df['INTERCEPT'] = model_dict[key][5][0]
        for i in range(len(predictor_names)):
           forecast_df[predictor_names[i] + '_coeff'] = model_dict[key][5][1][i] 
           
# 
#         #populate intercept and coefficient standard error columns in forecast_df with model values   
#         forecast_df['INTERCEPT_se'] = model_dict[key][9][0]
#         for i in range(len(predictor_names)):
#            forecast_df[predictor_names[i] + '_coeff_se'] = model_dict[key][9][1][i] 
#            
#         
#         #populate intercept and coefficient p value columns in forecast_df with model values  
#         forecast_df['INTERCEPT_p'] = model_dict[key][10][0]
#         for i in range(len(predictor_names)):
#            forecast_df[predictor_names[i] + '_p'] = model_dict[key][10][1][i] 
#            
# 
        
        #populate test_root_MSE from model test data
        forecast_df['test_root_MSE'] = model_dict[key][7]
    
        #populate R2 from model test data
        forecast_df['R2'] = model_dict[key][8]
        
#=============================================================================
    #Forecast Prediction for Model Gamma.  Gamma is the function definition
    #in the model Module. It is ok if it warns undefined
    
    #if model_dict[key][0] == Gamma:
    
    if model_dict[key][0] in [Gamma, NegativeBinomial,
                               generalizePoisson,generalizedPoisson2]:
         
        predictor_names = model_dict[key][6]
        #predictor_names = ['IFR', 'AWND', 'PRCP_SQRT', 'TMAX', 'isAHOLIDAY', 'SNOW_SQRT']
        
        #create x value DataFrame for predictions
        x_forecast = forecast_df[predictor_names]
        
        
        #Use model object from model_dict to forecast current plus next 7 days
        
        #pull model object from model_dict and name it results
        results = model_dict[key][4]
        
        #predict future VFR flights with saved model object from model_dict
        #if Gamma or NegativeBinomial
        if model_dict[key][0] in [Gamma, NegativeBinomial]:
            
            #predict future VFR flights with saved model object from model_dict
            poisson_predictions = results.get_prediction(x_forecast)
            predictions_summary_frame = poisson_predictions.summary_frame()
            print(key)
            print(predictions_summary_frame.head(15))
            
            #get future predictions of VFR Flight volume
            y_forecast = predictions_summary_frame['mean']
            
            #place future predictions of VFR in forecast_df
            forecast_df['y_forecast'] = y_forecast
            
            #pull confidence intervals for future predicted VFR and populate
            #forecast_df       
            forecast_df['CONF_lower'] = predictions_summary_frame['mean_ci_lower']
            
            forecast_df['CONF_upper'] = predictions_summary_frame['mean_ci_upper']
        
        
        #predict future VFR flights with saved model object from model_dict
        #if generalizePoisson,generalizedPoisson2
        
        if model_dict[key][0] in [generalizePoisson,generalizedPoisson2]:          
            
            y_forecast = results.predict(x_forecast)
            forecast_df['y_forecast'] = y_forecast
            
        #pull coefficients for parameters and populate forecast_df
        #(includes intercept for GLM models in main list)
        
        for i in range(len(predictor_names)):
            forecast_df[predictor_names[i] + '_coeff'] = results.params[i]
                       
        #pull Standard Errors of coefficients for parameters and populate
        #forecast_df (includes intercept for GLM models in main list)
        
        for i in range(len(predictor_names)):
            forecast_df[predictor_names[i] + '_coeff_se'] = results.bse[i]
                     
        #pull p values of coefficients for parameters and populate
        #forecast_df (includes intercept for GLM models in main list)
        
        for i in range(len(predictor_names)):
            forecast_df[predictor_names[i] + '_p'] = results.pvalues[i]
                     
        
        
    #--------------------------------------  
       
    forecast_dict[key] = [forecast_df]
    
    #-------------------------------------- 
    
    #add the current airport forecast_df to the combined forecast DataFrame
    combo_forecast_df = combo_forecast_df.append(forecast_df)
    
#make index a column with header day_index
combo_forecast_df.reset_index(inplace=True)
#give the day_index column correct column name
combo_forecast_df = combo_forecast_df.rename(columns = {'index':'day_index'})

    
#save csv for all combined airport forecast_df's
 
#get current working directory    
cwd = os.getcwd()
#identify path as the current working directory
path = cwd + "\\combo_forecast.csv"
combo_forecast_df.to_csv(path)

    
#--------------------------------------------

#create DataFrame and write csv to disk for point shape file
#do not have an older version of csv open on you computer or it won't write


#start lists to create DataFrame with shape file information
LOC_sh = []
LATITUDE_sh = []
LONGITUDE_sh = []
VFR_0 = []
VFR_1 = []
VFR_2 = []
VFR_3 = []
VFR_4 = []
VFR_5 = []
VFR_6 = []
VFR_7 = []

for key in forecast_dict:
    
    LOC_sh.append(forecast_dict[key][0]["LOC"][0])
    
    LATITUDE_sh.append(forecast_dict[key][0]["LATITUDE"][0])
     
    LONGITUDE_sh.append(forecast_dict[key][0]["LONGITUDE"][0])
     
    VFR_0.append(forecast_dict[key][0]["y_forecast"][0])
    
    VFR_1.append(forecast_dict[key][0]["y_forecast"][1])
    
    VFR_2.append(forecast_dict[key][0]["y_forecast"][2])
    
    VFR_3.append(forecast_dict[key][0]["y_forecast"][3])
    
    VFR_4.append(forecast_dict[key][0]["y_forecast"][4])
    
    VFR_5.append(forecast_dict[key][0]["y_forecast"][5])
    
    VFR_6.append(forecast_dict[key][0]["y_forecast"][6])
    
    VFR_7.append(forecast_dict[key][0]["y_forecast"][7])
    

# create shape dictionary of column lists 
shape_dict = {'LOC': LOC_sh, 'LATITUDE': LATITUDE_sh, 'LONGITUDE': LONGITUDE_sh,
             "VFR " + str(forecast_dict[key][0]["Date"][0]) : VFR_0,
             "VFR " + str(forecast_dict[key][0]["Date"][1]) : VFR_1,
             "VFR " + str(forecast_dict[key][0]["Date"][2]) : VFR_2,
             "VFR " + str(forecast_dict[key][0]["Date"][3]) : VFR_3,
             "VFR " + str(forecast_dict[key][0]["Date"][4]) : VFR_4,
             "VFR " + str(forecast_dict[key][0]["Date"][5]) : VFR_5,
             "VFR " + str(forecast_dict[key][0]["Date"][6]) : VFR_6,
             "VFR " + str(forecast_dict[key][0]["Date"][7]) : VFR_7 } 

#create DataFrame from dictionary    
shape_df = pd.DataFrame(shape_dict)
  

  
#write the shape_df to a shape_df.csv 
#do not have an older version of csv open on you computer or it won't write  
   
#get current working directory    
cwd = os.getcwd()
#identify path as the current working directory
path = cwd + "\\shape.csv"
shape_df.to_csv(path)


#print(forecast_df.dtypes)    





