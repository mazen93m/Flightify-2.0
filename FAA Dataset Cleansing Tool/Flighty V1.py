#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
os.getcwdb()


# In[5]:


# Import the files and Name them (rewrite directory path)

import numpy as np
import pandas as pd

def read_csv(csvfile,skiprows=0,header=0):
    
    return pd.read_csv(csvfile, skiprows=skiprows, header=header)

data = read_csv('data.csv')

#Check data imported
data.head(5)


# In[6]:


#Check datatypes
data.dtypes


# In[7]:


# Check for null values
print(data.info())

# Check for outliers
print(data.describe())

# IQR
Q1 = data.quantile(.25)
Q3 = data.quantile(.75)
IQR = Q3 - Q1
print(IQR)


# In[8]:


# Upper and Lower Limit
lower_lim = Q1 - 1.5 * IQR
upper_lim = Q3 + 1.5 * IQR
print('Lower Limit:', lower_lim, 'Upper Limit:', upper_lim)

# (Find a way to remove the outliers while maintaining as much data)


# In[9]:


data.hist(figsize=(10,10),bins=10)


# In[10]:


# Drop Columns
data = data.drop(['Date'], axis = 1)
data.head(5)


# In[11]:


#Data Preparation
data_cat = pd.get_dummies(data, drop_first = True)
data_cat.head(5)
data_cat.dtypes.head(30)


# In[12]:


data_cat.dtypes
data_cat = data_cat.rename(columns={"Holiday_New Year's Day": 'Holiday_New Years Day', "Holiday_New Year's Eve": 'Holiday_New Years Eve',
                              "Holiday_Valentine's Day": 'Holiday_Valentines Day', "Holiday_Washington's Birthday": 'Holiday_Washington Birthday'})
data_cat.dtypes


# In[13]:


# Split Data

from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(data_cat, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[14]:


# Re-scaling the features

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['Holiday_Christmas Day', 'Holiday_Christmas Eve', 'Holiday_Columbus Day', 'Holiday_Eastern Easter',
            'Holiday_Juneteenth','Holiday_Labor Day','Holiday_Labor Day Weekend',
           'Holiday_Martin Luther King, Jr. Day','Holiday_Memorial Day', 'Holiday_New Years Day',
          'Holiday_New Years Eve', 'Holiday_None','Holiday_Thanksgiving Day', 'Holiday_Thanksgiving Eve',
           'Holiday_Valentines Day','Holiday_Veterans Day','Holiday_Washington Birthday',
           'Holiday_Western Easter']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train
df_train.dtypes


# In[15]:


# Build Linear Model - All variables

y_train = df_train.pop('IFR')
x_train = df_train


# In[16]:


import statsmodels.api as sm
X_train_lm = sm.add_constant(x_train)

model_1 = sm.OLS(y_train, X_train_lm).fit()

model_1.summary()


# In[17]:


# VIR - Varience Inflation Factor
# how much feature variables are corrlated with each other

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = x_train.columns
vif['VIF'] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# Value greater than 5 is considered


# In[18]:


# Drop variables

X = x_train.drop(['LATITUDE'], axis= 1)

X = x_train.drop(['LONGITUDE'], axis= 1)

X = x_train.drop(['AWND'],axis= 1)

X = x_train.drop(['PRCP'],axis= 1)
X = x_train.drop(['SNOW'], axis= 1)
X = x_train.drop(['Holiday_Veterans Day'], axis= 1)
X = x_train.drop(['Holiday_Memorial Day'], axis= 1)
X = x_train.drop(['Holiday_New Years Eve'], axis= 1)
X = x_train.drop(['Holiday_Juneteenth'], axis= 1)
X = x_train.drop(['Holiday_Valentines Day'], axis= 1)
X = x_train.drop(['Holiday_Washington Birthday'],axis= 1)
X = x_train.drop(['Holiday_Thanksgiving Eve'],axis= 1)
X = x_train.drop(['Holiday_Eastern Easter'], axis= 1)
X = x_train.drop(['Holiday_New Years Day'], axis= 1)
X = x_train.drop(['Holiday_Labor Day'],axis= 1)
X = x_train.drop(['Holiday_Thanksgiving Day'], axis= 1)
X = x_train.drop(['Holiday_Christmas Day'], axis= 1)
X = x_train.drop(['Holiday_Columbus Day'], axis= 1)
X = x_train.drop(['SNWD'], axis= 1)
X = x_train.drop(['Holiday_Western Easter'], axis= 1)
X = x_train.drop(['Holiday_Labor Day Weekend'], axis= 1)
X = x_train.drop(['Holiday_Martin Luther King, Jr. Day'], axis= 1)


# In[19]:


X.dtypes


# In[20]:


# Build a fitted model 
X_train_lm = sm.add_constant(X)

model_2 = sm.OLS(y_train, X_train_lm).fit()

# Printing the summary of the model
print(model_2.summary())


# In[ ]:


# Residual analysis

