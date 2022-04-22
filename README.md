# Expanding Team Flightify: Predictable VFR Air Traffic Forecasting in All Major Airports in the US
Aircraft using VFR usually are not required to file a flight plan and are not automatically monitored by air traffic control. More resources are required to manage their flights which can cause a safety risk when airports are caught unprepared by a large influx of VFR traffic. Therefore, Team Flightify 2.0 will expand the work of the Summer 2021 Team Flightify to improve the accuracy of VFR flight predictions in class G airspace, which will enable the FAA to assess risk and allocate proper resources. The partner will be able to integrate the predictions into the ArcGIS (GLARE) platform. 

## Installation
#### Python environment

##### Run Data Cleaning Tool\
  Required files in current directory:\
    airports.py\
    FAA csv datasets\
    NOAA csv datasets
    
Run Model Module\
  Required files in current directory:\
    glm_modeling_scale.py\
    datasets.pkl

Run Forecast Module\
  Required files in current directory:\
    Pull_Forecast_api_openweather.py\
      Make sure API code is input on line #\
    model_dict.pkl\
    near_weekend_holiday_dates.csv
    
Using Tableau Dashboard\
  Required files in current directory:\
    Tableau Dashboard\
    combo_forecast.csv
    
Run Shape File Module:\
  Required files in current directory:\
    shape.csv

Instructions for aquiring openweathermap free API key code:\

1) Go to website: https://openweathermap.org/api \
2) Click Subscribe under One Call API\
   (API doc shows excellent explanation of all fields in API response)\
3) Choose "Get API Key" Under Free column for 1000 calls/day and 30,000 calls/month\  

## Usage

If historic data is updated, then begin with the Data Cleaning Tool and follow the installation order.
If no change in historic data, begin with the Forecast Module, then proceed to Tableau Dashboard, then Shape File Module.

```python
# execution example:
```
## Credits
George Mason Data Analytics Engineering Program: DAEN 690\
Spring 2022 Team Flightify 2.0: Dr. Charles Howard, Sabitha Pongadan, Alec Gray, Nida Sharief, Mazen Mohamed, Kimberly Cawi, Vasanthi Pulusu, Bharat Kumar Challakonda\
Summer 2021 Team Flightify: Deanna Snellings, Walter Benitez, Brittany Burwell, Jason Chern

## License


