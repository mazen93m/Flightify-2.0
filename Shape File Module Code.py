#!/usr/bin/env python
# coding: utf-8

# In[39]:


#Install packages if necessary
#!pip install wheel
#!pip install pipwin
#!pipwin install numpy
#!pipwin install pandas
#!pipwin install shapely
#!pipwin install gdal
#!pipwin install fiona
#!pipwin install pyproj
#!pipwin install six
#!pipwin install rtree
#!pipwin install geopandas

#Import necessary libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from fiona.crs import from_epsg


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
#Put in the directory where the VFR prediction output are

directory = 'C:/Users/spong/Desktop/GMUCourses/1DAEN690/Dummy'
os.listdir(directory)

for file in os.listdir(directory):
    if file.endswith(".csv"):
        name = os.path.splitext(file)[0]
        print(name)
        df = pd.read_csv(os.path.join(directory, file))

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

