# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:09:57 2022

@author: spong
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:33:47 2022

@author: spong
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from fiona.crs import from_epsg
import csv

#Define function to create point shapefile in WGS84format. The epsg code for wgs1984 is 4326. 

def convert_to_shapefile(airport_gdf, out_file):
    airport_gdf.crs = from_epsg(4326)
    airport_gdf.to_file(out_file)
    

#Define function to create circle shapefile in WGS84 format
#Buffer distance in decimal degrees, 0.725 corresponds to 50 miles

def convert_to_circleshapefile(airport_gdf, out_file):
      
        airport_gdf['geometry'] = airport_gdf['geometry'].buffer(.725)
        airport_gdf.crs = from_epsg(4326)
        airport_gdf.to_file(out_file)

        
#Code to loop through all files in a directory and create point and circle shapefiles

#get current working directory    
directory = os.getcwd()

os.listdir(directory)

for file in os.listdir(directory):
    if file.endswith(".csv"):
        name = os.path.splitext(file)[0]
        print(name)
        
#Shapefile CSV output from forecasting has an index with no header 
#which shows up in ArcGIS file instead of location while clicking a airport.
#To display the location ID correctly in ArcGIS, following code is used. 
        
        a=os.path.join(directory,file)
        with open(a) as infile:
            reader = csv.reader(infile)
            headers = next(reader)
            header_indices = [i for i, item in enumerate(headers) if item]
            df = pd.read_csv(a, usecols=header_indices)
   

#Rounding VFR Columns
            cols = ['VFR_Day_0','VFR_Day_1', 'VFR_Day_2', 'VFR_Day_3',
                    'VFR_Day_4', 'VFR_Day_5', 'VFR_Day_6','VFR_Day_7']
            df[cols] = df[cols].round(0)
            
                     
#Convert to GeoDataFrame. 
#Column names should be LATITUDE and LONGITUDE
        
        while True:
            try:
                airport_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['LONGITUDE'],
                                                                         df['LATITUDE']))
                break
            except KeyError:
                    print("Column names of Latitude and Longitude must be the same as in code")
                    break
                     
        point_filename= name + ".pointshape"
        point_file = os.path.join(directory, point_filename)
        convert_to_shapefile(airport_gdf, point_file)
        
        
        circle_filename= name + ".cirshape"
        circle_file = os.path.join(directory, circle_filename)
        convert_to_circleshapefile(airport_gdf, circle_file)