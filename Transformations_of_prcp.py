# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:15:19 2022
DAEN 690-004 Spring 2022 GMU
@author: Kimberly Cawi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

#--------------------------------------------------------------------------

#Show what the PRCP histogram BEFORE transformation looks like

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')

#plot histogram
FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')                 
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)


#-------------------------------------------------------------------------

#Quantile Transformer and Per Transformer Code from link below
#https://yashowardhanshinde.medium.com/what-is-skewness-in-data-how-to-fix-skewed-data-in-python-a792e98c0fa6

# This code does not all work.  Had to adjust code.


#cols1 = ["AWND", "PRCP","SNOW", "TMAX"]
#def test_transformers(columns):
#    pt = PowerTransformer()
#    qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')
#    fig = plt.figure(figsize=(20,30))
#    j = 1
#    for i in columns:
#        array = np.array(FAI_Noah_df[i]).reshape(-1, 1)
#        y = pt.fit_transform(array)
#        x = qt.fit_transform(array)
#        plt.subplot(3,3,j)
#        sns.histplot(array, bins = 50, kde = True)
#        plt.title(f"Original Distribution for {i}")
#        plt.subplot(3,3,j+1)
#        sns.histplot(x, bins = 50, kde = True)
#        plt.title(f"Quantile Transform for {i}")
#        plt.subplot(3,3,j+2)
#        sns.histplot(y, bins = 50, kde = True)
#        plt.title(f"Power Transform for {i}")
#        j += 4
#test_transformers(cols1)

#--------------------------------------------------------------------------

# Use Quantile transformer to transform the PRCP variable

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')

qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')

#make an array reshape to infer rows and force 1 column
array = np.array(FAI_Noah_df['PRCP']).reshape(-1,1)
x = qt.fit_transform(array)

#see what array looks like
x[:,0]

#place array values in PRCP
FAI_Noah_df['PRCP'] = x[:,0]

#plot histogram
FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')                 
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)



# Use Power transformer to transform the PRCP variable

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')

pt = PowerTransformer()

#make an array reshape to infer rows and force 1 column
array = np.array(FAI_Noah_df['PRCP']).reshape(-1,1)
x = pt.fit_transform(array)

#see what array looks like
x[:,0]

#place array values in PRCP
FAI_Noah_df['PRCP'] = x[:,0]

#plot histogram
FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')                 
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)



#---------------------------------------------------------------------------

# Transform PRCP sqrt.

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')

FAI_Noah_df['PRCP'] = FAI_Noah_df['PRCP']**(1./2)

FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)

#----------------------------------------------------------------------------

#Transform to PRCP **2

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')

FAI_Noah_df['PRCP'] = FAI_Noah_df['PRCP']**(2)

FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)

#----------------------------------------------------------------------------

# Transform to PRCP cube root.

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')


FAI_Noah_df['PRCP'] = FAI_Noah_df['PRCP']**(1./3)
FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)

#-----------------------------------------------------------------------------

# Transform to PRCP**(1/4).

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')


FAI_Noah_df['PRCP'] = FAI_Noah_df['PRCP']**(1./4)
FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)



#--------------------------------------------------------------------------
#  Transform PRCP log(PRCP + .0001)

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')

#  first replace 0 with .0001

#for i in FAI_Noah_df['PRCP']:
#  FAI_Noah_df["PRCP"].replace({0: .0001}, inplace=True)

FAI_Noah_df['PRCP'] = np.log(FAI_Noah_df['PRCP'] + .0001)
FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)


#----------------------------------------------------------------------------

#  Transform PRCP log(PRCP + 1)

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')

#  first replace 0 with .0001

#for i in FAI_Noah_df['PRCP']:
#  FAI_Noah_df["PRCP"].replace({0: .0001}, inplace=True)

FAI_Noah_df['PRCP'] = np.log(FAI_Noah_df['PRCP'] + 1)
FAI_Noah_df['PRCP'].plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)

#---------------------------------------------------------------------------

#Normalize PRCP to max min

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')


PRCP_scaled =(FAI_Noah_df['PRCP'] - FAI_Noah_df['PRCP'].min(axis=0) )/(FAI_Noah_df['PRCP'].max(axis=0) - FAI_Noah_df['PRCP'].min(axis=0))


PRCP_scaled.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('Precipitation')
plt.xlabel('Inches')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)



#---------------------------------------------------------------------------

#Example of BoxCox transformation using generated data - not our data
#https://www.geeksforgeeks.org/box-cox-transformation-using-python/

# Python3 code to show Box-cox Transformation 
# of non-normal data  (generated data for example)
  
# import modules

#import numpy as np
#from scipy import stats
  
# plotting modules

#import seaborn as sns
#import matplotlib.pyplot as plt
  
# generate non-normal data (exponential)

#original_data = np.random.exponential(size = 1000)

# transform training data & save lambda value

#fitted_data, fitted_lambda = stats.boxcox(original_data)
  
# creating axes to draw plots

#fig, ax = plt.subplots(1, 2)
  
# plotting the original data(non-normal) and 
# fitted data (normal)

#sns.distplot(original_data, hist = False, kde = True,
#          kde_kws = {'shade': True, 'linewidth': 2}, 
#          label = "Non-Normal", color ="green", ax = ax[0])
  
#sns.distplot(fitted_data, hist = False, kde = True,
#            kde_kws = {'shade': True, 'linewidth': 2}, 
#            label = "Normal", color ="green", ax = ax[1])
  
# adding legends to the subplots

#plt.legend(loc = "upper right")

# rescaling the subplots

#fig.set_figheight(5)
#fig.set_figwidth(10)
  
#print(f"Lambda value used for Transformation: {fitted_lambda}")

#-------------------------------------------------------------------


#Transform PRCP with BoxCox transformation
  
# import modules
import numpy as np
from scipy import stats
  
# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

FAI_Noah_df = pd.read_csv(
        'FAI_USW00026411_Noah_01012017_12312021_all temps_order2895112.csv')

#  first replace 0 with .0001

for i in FAI_Noah_df['PRCP']:
     FAI_Noah_df["PRCP"].replace({0: .0001}, inplace=True)
  
# generate non-normal data (exponential)
original_data = FAI_Noah_df['PRCP']

# transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(original_data)
  
# creating axes to draw plots
fig, ax = plt.subplots(1, 2)
  
# plotting the original data(non-normal) and 
# fitted data (normal)
sns.distplot(original_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2}, 
            label = "Non-Normal", color ="green", ax = ax[0])
  
sns.distplot(fitted_data, hist = False, kde = True,
            kde_kws = {'shade': True, 'linewidth': 2}, 
            label = "Normal", color ="green", ax = ax[1])
  
# adding legends to the subplots
plt.legend(loc = "upper right")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)
  
print(f"Lambda value used for Transformation: {fitted_lambda}")

