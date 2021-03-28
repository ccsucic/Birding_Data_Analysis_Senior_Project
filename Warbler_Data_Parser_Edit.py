# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:55:34 2020

@author: ccsuc
"""

#Importing necessary libraries
import pandas as pd
import glob
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
from sklearn.metrics import r2_score
from pathlib import Path


#------------------------------------------------
#        Combining eBird Warbler CSVs
#------------------------------------------------


path = r"C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/eBird_Warbler_Files_CSV"
all_files = glob.glob(path + "/*.csv")

#This is needed for us to append our looped data
li = []

#This loops over each file in our file folder
for filename in all_files:
    #Reads in CSV to a dataframe
    df = pd.read_csv(filename, low_memory = False, index_col = [0])
    #Drops columns that are not useful
    df_drop = df.drop(columns = ['LAST EDITED DATE','TAXONOMIC ORDER','CATEGORY','SUBSPECIES COMMON NAME',
                                  'SUBSPECIES SCIENTIFIC NAME','BREEDING BIRD ATLAS CODE','BREEDING BIRD ATLAS CATEGORY',
                                  'AGE/SEX','COUNTRY','COUNTRY CODE','IBA CODE','BCR CODE','USFWS CODE',
                                  'ATLAS BLOCK','LOCALITY','LOCALITY ID','LOCALITY TYPE','LATITUDE','LONGITUDE',
                                  'OBSERVER ID','SAMPLING EVENT IDENTIFIER','PROJECT CODE','HAS MEDIA','REASON', 
                                  'TRIP COMMENTS','SPECIES COMMENTS'])
    #Sets a variable that holds rows that have a value of X for the column OBSERVATION COUNT
    presence_only = df_drop[df_drop['OBSERVATION COUNT'] == 'X'].index
    #Drops rows that have a value of X for the column OBSERVATION COUNT
    df_drop.drop(presence_only, inplace = True)
    #Sets a variable that holds rows that have a 0 value for the column APPROVED
    indices = df_drop[df_drop['APPROVED'] == 0].index
    #Drops rows that have a 0 value for the column APPROVED
    df_drop.drop(indices, inplace = True)
    #Sets a dataframe where rows that have a NaN value in the column GROUP IDENTIFIER are dropped
    df_groupid_drop = df_drop.dropna(subset = ['GROUP IDENTIFIER'])
    #Sets a variable that contains T/F null values for the column GROUP IDENTIFIER
    filled_rows = df_drop['GROUP IDENTIFIER'].notnull()
    #Sets a variable that contains the rows of df_drop for which the value of filled_rows is True
    dropped_rows = df_drop[filled_rows].index
    #Sets a dataframe that contains the rows for which the column GROUP IDENTIFIER is null
    df_blank_groupid = df_drop.drop(dropped_rows)
    #Drops duplicate rows, considers all columns exact the first
    df_groupid_duplicate_drop = df_groupid_drop.drop_duplicates(subset = ['COMMON NAME','SCIENTIFIC NAME','OBSERVATION COUNT','STATE CODE',
                                              'OBSERVATION DATE','TIME OBSERVATIONS STARTED','PROTOCOL CODE','DURATION MINUTES',
                                              'EFFORT DISTANCE KM','EFFORT AREA HA','NUMBER OBSERVERS','ALL SPECIES REPORTED',
                                              'GROUP IDENTIFIER','APPROVED','REVIEWED'])
    #Duplicating the dataframe df_blank_groupid into df_combined
    df_combined = df_blank_groupid.copy()
    #Appends the dataframe df_groupid_duplicate_drop to df_combined
    df_combined = df_combined.append(df_groupid_duplicate_drop)
    #Resets the index of df_combined
    df_combined = df_combined.reset_index()
    #Drops an unneccesary named index
    df_combined.drop(columns = ['index'], inplace = True)
    #Making a copy of our dataframe in another variable
    df_testing = df_combined.copy()
    #Dropping rows where the duration minutes == NaN
    df_testing = df_testing.dropna(subset = ['DURATION MINUTES'])
    #Dropping rows where observations are incidental or random
    incidentals = df_testing[df_testing['PROTOCOL TYPE'] == 'Incidental'].index
    randoms = df_testing[df_testing['PROTOCOL TYPE'] == 'Random'].index
    df_testing.drop(incidentals, inplace = True)
    df_testing.drop(randoms, inplace = True)
    #Finding the rate of observations/minute by dividing the observation count by the duration minutes
    df_testing['Observations/Minute'] = df_testing['OBSERVATION COUNT'].astype(float) / df_testing['DURATION MINUTES']
    #Creating a new dataframe with only needed columns
    data = [df_testing['COMMON NAME'], df_testing['STATE'], df_testing['OBSERVATION DATE'], df_testing['Observations/Minute'], df_testing['OBSERVATION COUNT']]
    headers = ['COMMON NAME', 'STATE', 'OBSERVATION DATE', 'Observations/Minute', 'OBSERVATION COUNT']
    df_5col = pd.concat(data, axis = 1, keys = headers)
    #Converting the observation count to float for later calculations
    df_5col['OBSERVATION COUNT'] = df_5col['OBSERVATION COUNT'].astype(float)
    #Calculates the mean of observations/minute for matching values of common name, state, and observation date.
    #This gives us the average daily observations/minute for each species by state
    three_cols = ['COMMON NAME', 'STATE', 'OBSERVATION DATE']
    df_5col = df_5col.groupby(three_cols).mean().reset_index()
    #Sends our observation date column to datetime for reference later
    df_5col['OBSERVATION DATE'] = pd.to_datetime(df_5col['OBSERVATION DATE'])
    #Creates separate columns for the day, month, and year
    df_5col['Day'] = df_5col['OBSERVATION DATE'].dt.day
    df_5col['Month'] = df_5col['OBSERVATION DATE'].dt.month
    df_5col['Year'] = df_5col['OBSERVATION DATE'].dt.year
    #Drops uneccesary columns
    df_5col.drop(columns = ['OBSERVATION DATE', 'Day'], inplace = True)
    #Calculates the mean for observations/minute for matching values of common name, state, month, and year
    #This gives us the average monthly observations/minute for each species by state and year
    four_cols = ['COMMON NAME', 'STATE', 'Month', 'Year']
    df_5col = df_5col.groupby(four_cols).mean().reset_index()
    #Appending our values to a list for later dataframe building
    li.append(df_5col)
    
frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Warblers_Mean_original.csv")


#-----------------------------------------------------
#             Combining Drought CSVs
#-----------------------------------------------------


path = r"C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/Drought_Data"
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    #Getting the year from the name of the file
    name = Path(filename).stem
    year = int(name[10:14])
    #Reading in our drought data file
    df_drought = pd.read_csv(filename)
    #Drops StatisticFormatID column
    df_drought.drop(columns = ['StatisticFormatID'], inplace = True)
    #Creates list of columns to sum to find total area of drought in each state per week
    total_cols = ['D0', 'D1', 'D2', 'D3', 'D4']
    d1_cols = ['D1', 'D2', 'D3', 'D4']
    #Adds columns that sums total drought area of the state and total area of state above D1 drought level
    df_drought['Total_Drought_Area'] = df_drought[total_cols].sum(axis = 1)
    df_drought['D1_or_More'] = df_drought[d1_cols].sum(axis = 1)
    #Converts columns to datetime format for easier aggregation
    df_drought['MapDate'] = df_drought['MapDate'].astype(str)
    df_drought['MapDate'] = pd.to_datetime(df_drought['MapDate'])
    df_drought['ValidStart'] = pd.to_datetime(df_drought['ValidStart'])
    df_drought['ValidEnd'] = pd.to_datetime(df_drought['ValidEnd'])
    #Columns ValidStart and ValidEnd will be dropped for now, I may use them later
    df_drought.drop(columns = ['ValidStart', 'ValidEnd'], inplace = True)
    #This filters out any year not in the dataset
    df_drought = df_drought[df_drought['MapDate'].dt.year == year]
    #Averages the monthly percentage area under drought for each state
    columns = [pd.Grouper(key = 'MapDate', freq = 'M'), 'StateAbbreviation']
    df_drought = df_drought.groupby(columns).mean().reset_index()
    #Appending to our list for later use in concat
    li.append(df_drought)

#This creates a 
frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Drought_Data.csv")


#------------------------------------------------------
#          Combining Bird and Drought Data
#------------------------------------------------------


df_5col = pd.read_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Warblers_Mean_original.csv", low_memory = False, index_col = [0])
df_drought = pd.read_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Drought_Data.csv", low_memory = False, index_col = [0])
us_state_abbrev = {
'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}
df_5col['STATE'] = df_5col['STATE'].map(us_state_abbrev)
df_5col.rename(columns = {"STATE" : 'State'}, inplace = True)
df_drought.rename(columns = {"StateAbbreviation" : "State"}, inplace = True)
df_drought['MapDate'] = df_drought['MapDate'].astype(str)
df_drought['MapDate'] = pd.to_datetime(df_drought['MapDate'])
df_drought['Month'] = df_drought['MapDate'].dt.month
df_drought['Year'] = df_drought['MapDate'].dt.year
#This merges df_5col and df_drought based on the States and Month matching
df_5col = pd.merge(df_5col, df_drought, on=['State', 'Month', 'Year'], how='left')
df_5col.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Mean_Merge.csv")

