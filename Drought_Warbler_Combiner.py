import pandas as pd

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
warbler_combined_file = "Data/Combined_Warbler_CSV/02_to_19_Warblers_Mean_original.csv"
drought_combined_file = "Data/Combined_Drought_CSV/02_to_19_Drought_Data.csv"

def DroughtAndWarblerCombiner(drought_file, warbler_file):
    df_5col = pd.read_csv(warbler_file, low_memory = False, index_col = [0])
    df_drought = pd.read_csv(drought_file, low_memory = False, index_col = [0])
    df_5col['STATE'] = df_5col['STATE'].map(us_state_abbrev)
    df_5col.rename(columns = {"STATE" : 'State'}, inplace = True)
    df_drought.rename(columns = {"StateAbbreviation" : "State"}, inplace = True)
    df_drought['MapDate'] = df_drought['MapDate'].astype(str)
    df_drought['MapDate'] = pd.to_datetime(df_drought['MapDate'])
    df_drought['Month'] = df_drought['MapDate'].dt.month
    df_drought['Year'] = df_drought['MapDate'].dt.year
    #This merges df_5col and df_drought based on the States and Month matching
    df_5col = pd.merge(df_5col, df_drought, on=['State', 'Month', 'Year'], how='left')
    df_5col.to_csv("Data/Combined_Warbler_Drought_CSV/02_to_19_Mean_Merge.csv")