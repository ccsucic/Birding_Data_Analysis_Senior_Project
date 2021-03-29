import pandas as pd
import glob
from pathlib import Path

def DroughtDataCombiner(all_files):
    li_drought = []

    for filename in all_files:
        # Getting the year from the name of the file
        name = Path(filename).stem
        year = int(name[10:14])
        df_drought = pd.read_csv(filename)
        # Drops StatisticFormatID column
        df_drought.drop(columns = ['StatisticFormatID'], inplace = True)
        # Creates list of columns to sum to find total area of drought in each state per week
        total_cols = ['D0', 'D1', 'D2', 'D3', 'D4']
        d1_cols = ['D1', 'D2', 'D3', 'D4']
        # Adds columns that sums total drought area of the state and total area of state above D1 drought level
        df_drought['Total_Drought_Area'] = df_drought[total_cols].sum(axis = 1)
        df_drought['D1_or_More'] = df_drought[d1_cols].sum(axis = 1)
        # Converts columns to datetime format for easier aggregation
        df_drought['MapDate'] = df_drought['MapDate'].astype(str)
        df_drought['MapDate'] = pd.to_datetime(df_drought['MapDate'])
        df_drought['ValidStart'] = pd.to_datetime(df_drought['ValidStart'])
        df_drought['ValidEnd'] = pd.to_datetime(df_drought['ValidEnd'])
        # Columns ValidStart and ValidEnd will be dropped for now, I may use them later
        df_drought.drop(columns = ['ValidStart', 'ValidEnd'], inplace = True)
        # This filters out any year not in the dataset
        df_drought = df_drought[df_drought['MapDate'].dt.year == year]
        # Averages the monthly percentage area under drought for each state
        columns = [pd.Grouper(key = 'MapDate', freq = 'M'), 'StateAbbreviation']
        df_drought = df_drought.groupby(columns).mean().reset_index()
        # Appending to our list for later use in concat
        li_drought.append(df_drought)

    #This creates a data
    frame = pd.concat(li_drought, axis=0, ignore_index=True)
    frame.to_csv("Data/Combined_Drought_CSV/02_to_19_Drought_Data.csv")
