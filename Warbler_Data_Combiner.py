import pandas as pd
import glob

def WarblerDataCombiner(all_files):
    li_warblers = []
    for filename in all_files:
        df = pd.read_csv(filename, low_memory = False, index_col = [0])
        df_drop = df.drop(columns = ['LAST EDITED DATE','TAXONOMIC ORDER','CATEGORY','SUBSPECIES COMMON NAME',
                                      'SUBSPECIES SCIENTIFIC NAME','BREEDING BIRD ATLAS CODE','BREEDING BIRD ATLAS CATEGORY',
                                      'AGE/SEX','COUNTRY','COUNTRY CODE','IBA CODE','BCR CODE','USFWS CODE',
                                      'ATLAS BLOCK','LOCALITY','LOCALITY ID','LOCALITY TYPE','LATITUDE','LONGITUDE',
                                      'OBSERVER ID','SAMPLING EVENT IDENTIFIER','PROJECT CODE','HAS MEDIA','REASON', 
                                      'TRIP COMMENTS','SPECIES COMMENTS'])
        # Drops rows that have a value of X for the column OBSERVATION COUNT        
        presence_only = df_drop[df_drop['OBSERVATION COUNT'] == 'X'].index
        df_drop.drop(presence_only, inplace = True)
        # Drops rows that have a 0 value for the column APPROVED        
        indices = df_drop[df_drop['APPROVED'] == 0].index
        df_drop.drop(indices, inplace = True)
        # Sets a dataframe where rows that have a NaN value in the column GROUP IDENTIFIER are dropped
        df_groupid_drop = df_drop.dropna(subset = ['GROUP IDENTIFIER'])
        # Sets a variable that contains T/F null values for the column GROUP IDENTIFIER
        filled_rows = df_drop['GROUP IDENTIFIER'].notnull()
        # Sets a variable that contains the rows of df_drop for which the value of filled_rows is True
        dropped_rows = df_drop[filled_rows].index
        # Sets a dataframe that contains the rows for which the column GROUP IDENTIFIER is null
        df_blank_groupid = df_drop.drop(dropped_rows)
        # Drops duplicate rows, considers all columns exact the first
        df_groupid_duplicate_drop = df_groupid_drop.drop_duplicates(subset = ['COMMON NAME','SCIENTIFIC NAME','OBSERVATION COUNT',
                                                                              'STATE CODE','OBSERVATION DATE',
                                                                              'TIME OBSERVATIONS STARTED','PROTOCOL CODE',
                                                                              'DURATION MINUTES','EFFORT DISTANCE KM',
                                                                              'EFFORT AREA HA','NUMBER OBSERVERS','ALL SPECIES REPORTED',
                                                                              'GROUP IDENTIFIER','APPROVED','REVIEWED'])
        # Duplicating the dataframe df_blank_groupid into df_combined
        df_combined = df_blank_groupid.copy()
        # Appends the dataframe df_groupid_duplicate_drop to df_combined
        df_combined = df_combined.append(df_groupid_duplicate_drop)
        # Resets the index of df_combined
        df_combined = df_combined.reset_index()
        # Drops an unneccesary named index
        df_combined.drop(columns = ['index'], inplace = True)
        # Making a copy of our dataframe in another variable
        df_testing = df_combined.copy()
        # Dropping rows where the duration minutes == NaN
        df_testing = df_testing.dropna(subset = ['DURATION MINUTES'])
        # Dropping rows where observations are incidental or random
        incidentals = df_testing[df_testing['PROTOCOL TYPE'] == 'Incidental'].index
        randoms = df_testing[df_testing['PROTOCOL TYPE'] == 'Random'].index
        df_testing.drop(incidentals, inplace = True)
        df_testing.drop(randoms, inplace = True)
        # Finding the rate of observations/minute by dividing the observation count by the duration minutes
        df_testing['Observations/Minute'] = df_testing['OBSERVATION COUNT'].astype(float) / df_testing['DURATION MINUTES']
        # Creating a new dataframe with only needed columns
        data = [df_testing['COMMON NAME'], df_testing['STATE'], df_testing['OBSERVATION DATE'], df_testing['Observations/Minute'],
                df_testing['OBSERVATION COUNT']]
        headers = ['COMMON NAME', 'STATE', 'OBSERVATION DATE', 'Observations/Minute', 'OBSERVATION COUNT']
        df_5col = pd.concat(data, axis = 1, keys = headers)
        # Converting the observation count to float for later calculations
        df_5col['OBSERVATION COUNT'] = df_5col['OBSERVATION COUNT'].astype(float)
        # Calculates the mean of observations/minute for matching values of common name, state, and observation date.
        # This gives us the average daily observations/minute for each species by state
        three_cols = ['COMMON NAME', 'STATE', 'OBSERVATION DATE']
        df_5col = df_5col.groupby(three_cols).mean().reset_index()
        # Sends our observation date column to datetime for reference later
        df_5col['OBSERVATION DATE'] = pd.to_datetime(df_5col['OBSERVATION DATE'])
        # Creates separate columns for the day, month, and year
        df_5col['Day'] = df_5col['OBSERVATION DATE'].dt.day
        df_5col['Month'] = df_5col['OBSERVATION DATE'].dt.month
        df_5col['Year'] = df_5col['OBSERVATION DATE'].dt.year
        # Drops uneccesary columns
        df_5col.drop(columns = ['OBSERVATION DATE', 'Day'], inplace = True)
        # Calculates the mean for observations/minute for matching values of common name, state, month, and year
        # This gives us the average monthly observations/minute for each species by state and year
        four_cols = ['COMMON NAME', 'STATE', 'Month', 'Year']
        df_5col = df_5col.groupby(four_cols).mean().reset_index()
        # Appending our values to a list for later dataframe building
        li_warblers.append(df_5col)

    frame = pd.concat(li_warblers, axis=0, ignore_index=True)
    frame.to_csv("Data/Combined_Warbler_CSV/02_to_19_Warbler_Data.csv")