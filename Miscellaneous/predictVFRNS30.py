#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:39:25 2022
@author: ajgray
"""

import dataCleaner as d
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
import math

# reading in the data csv file outputed by the cleaning tool
data = d.datasets['ANC']

sns.set()
correlation_matrix = data[['VFR','AWND','SNOW','TMAX','isAHoliday','PRCP']].corr()

sns.heatmap(correlation_matrix)
plt.show()

sns.scatterplot(x='VFR',y='PRCP',data=data)
plt.xlabel('Daily VFR Traffic')
plt.ylabel('Daily Precipitation')
plt.title('Daily Precipitation by VFR Traffic')
plt.show()

sns.scatterplot(x='VFR',y='TMAX',data=data)
plt.xlabel('Daily VFR Traffic')
plt.ylabel('Daily Maximum Temperature')
plt.title('Maximum Temp by VFR Traffic')
plt.show()

data_np = data.to_numpy()

# dataset to train and test 
labels = data_np[:, -8:-1]

# dependent label variable
features = data_np[:, -9]
features = features.reshape(-1,1)

# Split into train and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 100)

# Training and Test Shape

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Train Model
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 50)
# Train the model on training data
rf.fit(train_features, train_labels);

print(f'model score on testing data: {rf.score(test_features, test_labels)}')
print(f'model score on testing data: {rf.score(train_features, train_labels)}')

# Make Predictions
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

from sklearn import metrics
# Error
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))

# Expanded Data
# Instantiate random forest and train on new features
from sklearn.ensemble import RandomForestRegressor
rf_exp = RandomForestRegressor(n_estimators= 1000, random_state=100)
rf_exp.fit(train_features, train_labels)

# Make predictions on test data
predictions = rf_exp.predict(test_features)
# Performance metrics
errors = abs(predictions - test_labels)
print('Metrics for Random Forest Trained on Expanded Data')
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')

# Get numerical feature importances

feature_list = list(features.columns)

importances = list(rf_exp.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
