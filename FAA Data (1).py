#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
os.getcwdb()


# In[5]:


# Import the files and Name them (rewrite directory path)

import pandas as pd
Census_Data = pd.read_csv("C:/Users/Naheed/Documents/DAE/Data/Censes_group_data_sources.csv")
FAA_Est_Data = pd.read_csv("C:/Users/Naheed/Documents/DAE/Data/FAA_estimated_state_level_hour.csv")
GA_Data = pd.read_csv("C:/Users/Naheed/Documents/DAE/Data/GA_survey.csv")
Region_Data = pd.read_csv("C:/Users/Naheed/Documents/DAE/Data/Region_state_Lookup.csv")
StateArea_Data = pd.read_csv("C:/Users/Naheed/Documents/DAE/Data/state_area.csv")
StateCode_Data = pd.read_csv("C:/Users/Naheed/Documents/DAE/Data/State_code.csv")
Zipcode_Data = pd.read_csv("C:/Users/Naheed/Documents/DAE/Data/ZIPCode_State_Lookup.csv")

print(Census_Data.head(5))
print(FAA_Est_Data.head(5))
print(GA_Data.head(5))
print(Region_Data.head(5))
print(StateArea_Data.head(5))
print(StateCode_Data.head(5))
print(Zipcode_Data.head(5))


# In[6]:


# Merge Tables

Census_Data.dtypes
df1 = pd.merge(StateArea_Data, StateCode_Data, on="STATE_1")


# In[7]:


df2 = pd.merge(df1, Region_Data, on="STATE")
df2 = df2.drop(columns=['STATE_1_y', 'State_code_y'])
df2 = df2.rename(columns={"STATE_1_x": "State", "STATE": "STATE_Abbr", "State_code_x": "State_code"})
df2.head(5)


# In[8]:


Zipcode_Data.dtypes
df3 = pd.merge(Zipcode_Data, df2, on="State")
df3.head(5)


# In[9]:


df3 = df3.drop(columns=['State_code_x'])
df3 = df3.rename(columns={"State_code_y": "State_code"})
df3 = df3.rename(columns={"Area": "State_Area"})
df3.head(5)


# In[10]:


df4 = pd.merge(df3, FAA_Est_Data, on="State")
df4 = df4.rename(columns={"Hours": "FAA_Est_Hours"})
df4.head(5)


# In[11]:


df4.head(5)


# In[12]:


Census_Data.head(5)


# In[13]:


Census_Data['ZipCode'] = pd.to_numeric(Census_Data['ZipCode'], errors='coerce')
Census_Data.dtypes


# In[14]:


Census_CData = pd.merge(Census_Data, df4, on="ZipCode")
Census_CData.head(5)


# In[15]:


print(GA_Data.head(5))
GA_Data = GA_Data.rename(columns={"REGION": "REGION_NAME", "Aircraft": "GA_Aircraft",
                                  "Forest": "GA_Forest","School": "GA_School","Airport": "GA_Airport",
                                  "Area": "GA_Area", " General Aviation Use Sight See": "General Aviation Use Sight See"})
GA_Data.dtypes
Region = [1,2,3,4,5,6,7,8,9]
GA_Data['Region']=Region
GA_Data = GA_Data.drop(columns=['Unnamed: 1', 'Unnamed: 5','Unnamed: 13','Unnamed: 19'])

GA_Data.head(10)


# In[16]:


Census_CData.dtypes


# In[17]:


GA_Data.dtypes


# In[18]:


GA_Census_Data = pd.merge(Census_CData, GA_Data, on='Region')
GA_Census_Data.head(5)


# In[19]:


GA_Census_Data.dtypes


# In[20]:


GA_Data.dtypes


# In[21]:


# Missing Data
GA_Census_Data.isna().sum()


# In[22]:


GA_Census_Data.dtypes


# In[23]:


# Convert for encoding
GA_Census_Data['All_hours'] = GA_Census_Data['All_hours'].replace(',','', regex=True)
GA_Census_Data.dtypes
GA_Census_Data['All_hours'] = GA_Census_Data['All_hours'].astype(int)
GA_Census_Data['All_hours'].head(5)
GA_Census_Data['FAA_Est_Hours'] = GA_Census_Data['FAA_Est_Hours'].replace(',','', regex=True)
GA_Census_Data['FAA_Est_Hours'] = GA_Census_Data['FAA_Est_Hours'].astype(int)
GA_Census_Data['FAA_Est_Hours'].head(5)


# In[24]:


# Visualization 

import seaborn as sns

# Plot the histogram thanks to the distplot function
sns.distplot( a=GA_Census_Data["FAA_Est_Hours"],
             hist=True, kde=False, rug=False).set(title='FAA Estimated Hours')


# In[25]:


# Plot the histogram thanks to the distplot function
sns.distplot( a=GA_Census_Data["GA_Area"],
             hist=True, kde=False, rug=False).set(title='GA_Area')


# In[91]:


# Plot the histogram thanks to the distplot function
sns.barplot(x="FAA_Est_Hours", y="REGION_NAME", data=GA_Census_Data).set(title='Estimated Flight Hours by Region')


# In[99]:


# Plot the histogram thanks to the distplot function
top_10 = (GA_Census_Data.groupby('State')['FAA_Est_Hours', 'PILOT',
                                          'Aircraft',
                                          'Airport'].agg({'FAA_Est_Hours': 'sum',
                                                                      'PILOT': 'count',
                                                          'Aircraft': 'count', 'Airport':'count'}).sort_values(by='FAA_Est_Hours', ascending=False))[:10].reset_index()
top_10


# In[102]:


sns.barplot(x="PILOT", y="State", data=top_10).set(title='Number of Pilots by State')


# In[104]:


sns.barplot(x="Aircraft", y="State", data=top_10).set(title='Number of Aircraft by State')


# In[106]:


sns.barplot(x="Airport", y="State", data=top_10).set(title='Number of Airport by State')


# In[94]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set(title='Estimated Flight Hours by State', xlabel='Estimated Flight Hours', ylabel='State')
top_10.plot(kind='barh', y="FAA_Est_Hours", x="State", ax=ax)


# In[24]:


# Plot the histogram thanks to the distplot function
sns.distplot( a=GA_Census_Data["All_hours"],
             hist=True, kde=False, rug=False).set(title='All hours')


# In[25]:


# Convert Categorical data
GAC_Data = pd.get_dummies(data=GA_Census_Data)
GAC_Data.dtypes


# In[26]:


# Array Manipulation

import numpy as np
Pred_FH = np.array(GAC_Data['FAA_Est_Hours'])
GACF_Data= GAC_Data.drop('FAA_Est_Hours', axis = 1)
feature_list = list(GACF_Data.columns)
features = np.array(GACF_Data)


# In[27]:


# Trainning and Test data

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_FH, test_FH = train_test_split(features, Pred_FH, 
test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Flight Hours Shape:', train_FH.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Flight Hours Shape:', test_FH.shape)


# In[28]:


# Train Model

from sklearn.ensemble import RandomForestRegressor
random_f = RandomForestRegressor(n_estimators = 100, random_state = 42)
random_f.fit(train_features, train_FH)


# In[29]:


# Make Predictions

# Use the forest's predict method on the test data
predictions = random_f.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_FH)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))


# In[30]:


# Calculate mean absolute percentage error

mape = 100 * (errors / test_FH)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[31]:


# Get numerical feature importances
importances = list(random_f.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:148} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[32]:


# New random forest with only the 5 most important variables
randf_most_important = RandomForestRegressor(n_estimators= 100, random_state=42)

# Extract the two most important features
important_indices = [feature_list.index('State_Area'), feature_list.index('State_Florida'),
                     feature_list.index('STATE_Abbr_FL'),feature_list.index('ZipCode'),
                     feature_list.index('State_code')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]

# Train the random forest
randf_most_important.fit(train_important, train_FH)

# Make predictions and determine the error
predictions = randf_most_important.predict(test_important)
errors = abs(predictions - test_FH)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2))
mape = np.mean(100 * (errors / test_FH))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# In[34]:


# Poisson Distribution - Probability mass function

from scipy.stats import poisson
import matplotlib.pyplot as plt
#
lmbda = GAC_Data['FAA_Est_Hours'].mean()
X = GAC_Data['FAA_Est_Hours']

poisson_pd = poisson.pmf(X, lmbda)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(X, poisson_pd, 'bo', ms=8, label='poisson pmf')
plt.ylabel("Probability", fontsize="18")
plt.xlabel("X - FAA Estimated Hours", fontsize="18")
plt.title("Poisson Distribution - No. of FAA Estimated Hours Vs Probability", fontsize="18")
ax.vlines(X, 0, poisson_pd, colors='b', lw=5, alpha=0.5)


# In[45]:


# Poisson Distribution - Cumulative density function

poisson_cdf = poisson.cdf(X, lmbda)

# Plot the probability distribution

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(X, poisson_cdf, 'bo', ms=8, label='poisson cdf')
plt.ylabel("Probability", fontsize="18")
plt.xlabel("X - FAA Estimated Hours", fontsize="18")
plt.title("Poisson Distribution - No. of FAA Estimated Hours Vs Probability", fontsize="18")
ax.vlines(X, 0, poisson_cdf, colors='b', lw=5, alpha=0.5)


# In[ ]:




