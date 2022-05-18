# Flightify 2.0 - 
# Expanding Team Flightify: Predictable VFR Air Traffic Forecasting in All Major Airports in the US
Aircraft using VFR usually are not required to file a flight plan and are not automatically monitored by air traffic control. More resources are required to manage their flights which can cause a safety risk when airports are caught unprepared by a large influx of VFR traffic. Therefore, Team Flightify 2.0 models and forecasts future VFR flight volume in class G airspace, which will enable the FAA to assess risk and allocate proper resources. The partner will be able to integrate the predictions into the ArcGIS (GLARE) platform.

## Installation
#### Python environment

##### Run Data Cleaning Tool
  Required files in current directory:\
  airports.py\
  dataCleaner.py\
  NOAA.zip files extracted\
  Tower_ops_airports.zip files extractedfaa_lat_long.csv\
  updated_holidays.csv (unused, but available)\

    
    
##### Run Model Module
  Required files in current directory:\
    glm_modeling_scale.py\
    datasets.pkl

##### Run Forecast Module
  Required files in current directory:\
    Pull_Forecast_api_openweather.py\
      Make sure API code is input on line #102\
      Execution output writes to a text file, so do not be alarmed when you see nothing happening on screen while it runs 5-7 minutes.
    model_dict.pkl\
    near_weekend_holiday_dates.csv
      has dates through 12/29/23
    
##### Using Tableau Dashboard
  Required files in current directory:\
    Tableau Dashboard\
    combo_forecast.csv\
    FAA Location Data excel (to collect location data for airports and automate the else/if statement code for Tableau Column calculation.)
    
##### Run Shape File Module
  Required files in current directory:\
    shape file module final.py\
    shape.csv


## Usage

#### Instructions for aquiring openweathermap free API key code:

1) Go to website: https://openweathermap.org/api 
2) Click Subscribe under One Call API 3.0\
   (API doc shows excellent explanation of all fields in API response)


Before using the system insert your open weather api code in the Pull_Forecast_api_openweather.py script in line 102.  See instructions below.

```python

#Insert api key in the code line in the script, Pull_Forecast_api_openweather.py line 102

#Important: Insert your API key in place of {api key}, drop braces, no spaces

openweather_api = 'https://api.openweathermap.org/data/3.0/onecall?exclude=hourly,alerts,minutely&appid={api key}&units=imperial'

```

Email for OpenWeather API:  info@openweathmap.org
They are very responsive to questions


-If historic data is updated, then begin with the Data Cleaning Tool and follow the installation order.\
-If no change in historic data, begin with the Forecast Module, then proceed to Tableau Dashboard, then Shape File Module.


#### ArcGIS Shapefile instructions
Open ArcGIS Pro app\
Select the Map Template\
Go to View tab -> Catalog Pane, and a catalog pane window pops up\
From there, go to 'Project' tab->Folders. Right click on 'Folders' tab, and select 'Add Folder Connection'\
Add the shape file folder. (Note: Each shape file will be in a folder when it is created)\
The new added folder can be seen under the Folders tab\
Double click on it, and the necessary file to create a layer in ArcGIS will be shown below the shape file folder\
Drag that file onto the map\
The shapefile will be plotted on the map\
Additional shape files can be added and opened onto the same map by repeating the steps above\
Once plotted on the map, changes can be made to the points/circle/geometry by either clicking on them or in the 'Feature Layer' -> 'Appearance' tab\


## Credits
George Mason Data Analytics Engineering Program: DAEN 690\
Spring 2022 Team Flightify 2.0: Dr. Charles Howard, Sabitha Pongadan, Alec Gray, Nida Sharief, Mazen Mohamed, Kimberly Cawi, Vasanthi Pulusu, Bharat Kumar Challakonda\
Summer 2021 Team Flightify: Deanna Snellings, Walter Benitez, Brittany Burwell, Jason Chern




