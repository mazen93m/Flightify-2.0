#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:21:05 2022

@author: ajgray
"""
import os
import pandas as pd
import csv
import pickle

# =============================================================================
# airports_dict = {'fai':['fai_opsnet_tower_ops_2017-2021.csv','fai_noaa_usw00026411_2017-2021.csv', 'fairbanks international airport, ak us'],
#              'anc':['anc_opsnet_tower_ops_2017-2021.csv','anc_noaa_usw00026451_2017-2021.csv', 'anchorage ted stevens international airport, ak us'],
#              'jnu':['jnu_opsnet_tower_ops_2017-2021.csv','jnu_noaa_usw00025309_2017-2021.csv', 'juneau airport, ak us']}
# =============================================================================

tower_ops_path = '/Users/ajgray/Desktop/project/tower_ops_airports'
tower_ops = os.listdir(tower_ops_path)
tower_ops.sort()

noaa_path = '/Users/ajgray/Desktop/project/NOAA'
noaa_weather = os.listdir(noaa_path)
noaa_weather.sort()

duplicate_airports = {'ADS': 'ADDISON AIRPORT, TX US',
 'ASG': 'SPRINGDALE MUNICIPAL AIRPORT, AR US',
 'ATL': 'HARTSFIELD-JACKSON ATLANTA INTERNATIONAL AIRPORT, GA US',
 'BDR': 'SIKORSKY MEMORIAL AIRPORT, CT US',
 'BWI': 'BALTIMORE/WASHINGTON INTERNATIONAL THURGOOD MARSHALL AIRPORT, MD US',
 'CRE': 'GRAND STRAND AIRPORT, SC US',
 'CWA': 'CENTRAL WISCONSIN AIRPORT, WI US',
 'CWF': 'CHENNAULT INTERNATIONAL AIRPORT, LA US',
 'CXO': 'CONROE-NORTH HOUSTON REGIONAL AIRPORT, TX US',
 'DAB': 'DAYTONA BEACH INTERNATIONAL AIRPORT, FL US',
 'DFW': 'DALLAS/FORT WORTH INTERNATIONAL AIRPORT, TX US',
 'DTO': 'DENTON ENTERPRISE AIRPORT, TX US',
 'DVT': 'PHOENIX DEER VALLEY AIRPORT, AZ US',
 'EAU': 'CHIPPEWA VALLEY REGIONAL AIRPORT, WI US',
 'ENW': 'KENOSHA REGIONAL AIRPORT, WI US',
 'FFZ': 'FALCON FIELD AIRPORT, AZ US',
 'FIN': 'FLAGLER EXECUTIVE AIRPORT, FL US',
 'FWS': 'FORT WORTH SPINS AIRPORT, TX US',
 'GTR': 'GOLDEN TRIANGLE REGIONAL AIRPORT, MS US',
 'GYI': 'NORTH TEXAS REGIONAL AIRPORT, TX US',
 'GYR': 'PHOENIX GOODYEAR AIRPORT, AZ US',
 'GYY': 'GARY/CHICAGO INTERNATIONAL AIRPORT, IN US',
 'HQZ': 'MESQUITE METRO AIRPORT, TX US',
 'IAH': 'GEORGE BUSH INTERCONTINENTAL AIRPORT, TX US',
 'ILM': 'WILMINGTON INTERNATIONAL AIRPORT, NC US',
 'IWA': 'PHOENIX-MESA GATEWAY AIRPORT, AZ US',
 'JAX': 'JACKSONVILLE INTERNATIONAL AIRPORT, FL US',
 'JVL': 'SOUTHERN WISCONSIN REGIONAL AIRPORT, WI US',
 'LAL': 'LAKELAND LINDER INTERNATIONAL AIRPORT, FL US',
 'LAX': 'LOS ANGELES INTERNATIONAL AIRPORT, CA US',
 'LEB': 'LEBANON MUNICIPAL AIRPORT, NH US',
 'LCH': 'LAKE CHARLES REGIONAL AIRPORT, LA US',
 'LOU': 'BOWMAN FIELD, KY US',
 'LWB': 'GREENBRIER VALLEY AIRPORT, WV US',
 'MDH': 'SOUTHERN ILLINOIS AIRPORT, IL US',
 'MEI': 'MERIDIAN KEY FIELD, MS US',
 'MEM': 'MEMPHIS INTERNATIONAL AIRPORT, TN US',
 'MHR': 'MATHER AIRPORT, CA US',
 'MSY': 'LOUIS ARMSTRONG NEW ORLEANS INTERNATIONAL AIRPORT, LA US',
 'MTN': 'MARTIN STATE AIRPORT, MD US',
 'MWA': 'VETERANS AIRPORT OF SOUTHERN ILLINOIS, IL US',
 'MYR': 'MYRTLE BEACH INTERNATIONAL AIRPORT, SC US',
 'NEW': 'LAKEFRONT AIRPORT, LA US',
 'NQA': 'MILLINGTON-MEMPHIS AIRPORT, TN US',
 'OAJ': 'ALBERT J ELLIS AIRPORT, NC US',
 'ORD': "O'HARE INTERNATIONAL AIRPORT, IL US",
 'ORL': 'ORLANDO EXECUTIVE AIRPORT, FL US',
 'OUN': 'MAX WESTHEIMER AIRPORT, OK US',
 'OWB': 'OWENSBORO-DAVIESS COUNTY REGIONAL AIRPORT, KY US',
 'OXC': 'WATERBURY-OXFORD AIRPORT, CT US',
 'PAH': 'BARKLEY REGIONAL AIRPORT, KY US',
 'PAO': 'PALO ALTO AIRPORT, CA US',
 'PHX': 'PHOENIX SKY HARBOR INTERNATIONAL AIRPORT, AZ US',
 'PIE': 'ST. PETE-CLEARWATER INTERNATIONAL AIRPORT, FL US',
 'PWA': 'WILEY POST AIRPORT, OK US',
 'RBD': 'DALLAS EXECUTIVE AIRPORT, TX US',
 'ROG': 'ROGERS MUNICIPAL AIRPORT, AR US',
 'RYN': 'RYAN BLUESON AIRFIELD, AZ US',
 'RYY': 'COBB COUNTY INTERNATIONAL AIRPORT, GA US',
 'SAC': 'SACRAMENTO INTERNATIONAL AIRPORT, CA US',
 'SBD': 'SAN BERNARDINO INTERNATIONAL AIRPORT, CA US',
 'SDF': 'LOUISVILLE INTERNATIONAL AIRPORT, KY US',
 'SFO': 'SAN FRANCISCO INTERNATIONAL AIRPORT, CA US',
 'SNS': 'SALINAS MUNICIPAL AIRPORT, CA US',
 'SQL': 'SAN CARLOS AIRPORT, CA US',
 'SUA': 'WITHAM FIELD, FL US',
 'TIX': 'SPACE COAST REGIONAL AIRPORT, FL US',
 'TKI': 'MCKINNEY NATIONAL AIRPORT, TX US',
 'TUP': 'TUPELO REGIONAL AIRPORT, MS US',
 'TUS': 'TUSCON INTERNATIONAL AIRPORT, AZ US',
 'VQQ': 'CECIL AIRPORT, FL US',
 'VRB': 'VERO BEACH REGIONAL AIRPORT, FL US',
 'XNA': 'NORTHWEST ARKANSAS NATIONAL AIRPORT, AR US'
 }

# =============================================================================
# for t_el in tower_ops:
#     for n_el in noaa:
#         if t_el[:3] not in :
# =============================================================================
        

def removeDSFile(lst):
    for el in lst:
        if '.DS' in el:
            lst.remove(el)
            
removeDSFile(tower_ops)
removeDSFile(noaa_weather)

#t_codes = [el[:3] for el in tower_ops]
#n_codes = [el[:3] for el in noaa_weather]

#n_set = set(n_codes)
#t_set = set(t_codes)

#tower = set(tower_ops)
#noaa = set(noaa_weather)

def getDF(csv):
    os.chdir(noaa_path)
    noaaFrame = pd.read_csv(csv, header=0)    
    
    return noaaFrame

def getName(df):
    os.chdir(noaa_path)
    noaaFrame = pd.read_csv(df, header=0)
    name = noaaFrame.loc[0]['NAME']
    
    return name 

airports = {}

def changeName(csv, name):
    df = getDF(csv)
    df.NAME = name
    
    #print(df.head())
    
    return name

def createAirportDict():
    for i in range(len(tower_ops)):
        if tower_ops[i][:3].upper() in duplicate_airports:
            #print('{}: {}\n'.format(noaa_weather[i],missing_airports[tower_ops[i][:3].upper()]))
            airports[tower_ops[i][:3].upper()] = [tower_ops[i],noaa_weather[i],changeName(noaa_weather[i],duplicate_airports[tower_ops[i][:3].upper()])]
        else:
            airports[tower_ops[i][:3].upper()] = [tower_ops[i],noaa_weather[i],getName(noaa_weather[i])]
    
    return airports

def renameAirports():
    for key in airports:
        if key in duplicate_airports:
            airports[key][2] = duplicate_airports[key]
            
createAirportDict()
renameAirports()

def createDictFile():
    # open file for writing, "w" is writing
    w = csv.writer(open("output.csv", "w"))
    
    # loop over dictionary keys and values
    for key, val in airports.items():
    
        # write every key and value to file
        w.writerow([key, val])
        
#createDictFile()
