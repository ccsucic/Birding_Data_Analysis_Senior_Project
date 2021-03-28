# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:39:01 2020

@author: ccsuc
"""
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
from sklearn.metrics import r2_score


#-----------------------------------------------------------
#      Linear Regression on State-Month, Individual Species
#-----------------------------------------------------------


r2_values = []
state_list = ['AL', 'GA', 'FL', 'LA', 'MS', 'NC', 'SC', 'VA']
species = ["Ovenbird","Worm-eating Warbler","Louisiana Waterthrush",
            "Northern Waterthrush","Golden-winged Warbler","Blue-winged Warbler",
            "Black-and-white Warbler","Prothonotary Warbler","Swainson's Warbler",
            "Tennessee Warbler","Orange-crowned Warbler","Nashville Warbler",
            "Connecticut Warbler","Mourning Warbler","Kentucky Warbler",
            "Common Yellowthroat","Hooded Warbler","American Redstart","Cape May Warbler",
            "Cerulean Warbler","Northern Parula","Magnolia Warbler","Bay-breasted Warbler",
            "Blackburnian Warbler","Yellow Warbler","Chestnut-sided Warbler",
            "Blackpoll Warbler","Black-throated Blue Warbler","Palm Warbler","Pine Warbler",
            "Yellow-rumped Warbler","Yellow-throated Warbler","Prairie Warbler",
            "Black-throated Green Warbler","Canada Warbler","Wilson's Warbler"]
droughts = ['Total_Drought_Area', 'D1_or_More', 'D2_or_More', 'D3_or_More']
for drought in droughts:
    for state in state_list:
        for i in range(1, 13):
            for bird in species:
                df_merge = pd.read_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Mean_Merge.csv", low_memory = False, index_col = [0])
                df = df_merge[(df_merge['Month'] == i) & (df_merge['State'] == state) & (df_merge['COMMON NAME'] == bird) & (df_merge['Year'] >= 2008)].copy()
                print(df['Year'])
                d2_cols = ['D2', 'D3', 'D4']
                d3_cols = ['D3', 'D4']
                df["D2_or_More"] = df[d2_cols].sum(axis = 1)
                df["D3_or_More"] = df[d3_cols].sum(axis = 1)
                #print(df)
                df.replace([np.inf, -np.inf], np.nan, inplace = True)
                df.dropna(inplace = True)
                if len(df) < 10:
                    print(state, i, bird)
                    continue
                #print(df)
                #We don't need this section since we're not dealing with all warblers
                # drops = ['COMMON NAME', 'MapDate', 'State']
                # df.drop(columns = drops, inplace = True)
                # columns = ['Year', 'Month']
                # df = df.groupby(columns).mean()
                # #print(df)
                #Defining the independent and dependent variables for regression
                # X = df[drought].values.reshape(-1, 1)
                y = df["Observations/Minute"].values.reshape(-1, 1)
                #Running the linear regression
                lm = linear_model.LinearRegression()
                lm.fit(X,y)
                y_pred = lm.predict(X)
                #Finding the P Value
                X_ = sm.add_constant(X)
                model = sm.OLS(y,X_)
                results = model.fit()
                #Finding Correlation Coefficient (R Value) using pandas
                pear_corr = df[drought].corr(df["Observations/Minute"])
                years = ['2010', '2011', '2012', '2013', 
                          '2014', '2015', '2016', '2017', '2018', '2019']
                #Plotting section            
                plt.scatter(X, y)
                plt.plot(X, y_pred, color = "red")
                title = "Linear Regression: FL, 8, Blackburnian Warbler"
                plt.title(title)
                plt.xlabel("D3_or_More")
                plt.ylabel("Observations/Minute")
                annotation = 'R^2 = ', r2_score(y, y_pred)
                plt.annotate(annotation, xy = (6, 0.01))
                pval = 'P-Val = ', results.pvalues[1]
                plt.annotate(pval, xy = (6, 0.00))
                for j,txt in enumerate(years):
                    plt.annotate(txt, (X[j], y[j]))
                plt.show()
                
                #Finding mean of residuals
                print(results.resid.mean())
                #Attaching R^2 to a list for a later dataframe
                r2_values.append(state)
                r2_values.append(i)
                r2_values.append(bird)
                r2_values.append(r2_score(y, y_pred))
                r2_values.append(results.pvalues[1])
                r2_values.append(pear_corr)
                r2_values.append(drought)
                print(state, i, bird, r2_score(y, y_pred), results.pvalues[1], pear_corr, drought)

#print(r2_values)
df_r2 = pd.DataFrame(np.array(r2_values).reshape(-1, 7), columns = ['State', 'Month', 'Species', 'R^2 Value', 'P Value', 'R value/Correlation Coeff', 'Drought Level'] )
#print(df_r2)
df_r2.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/State Month Species Mean R2 Val and P Val All Droughts.csv")
#df.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/All Warblers Linear Reg.csv")

##############################################################
#        Code not used in project beyond this point
##############################################################


# #Making the Spring/Fall significance chart
# li = []
# for bird in species:
#     for drought in droughts:
#         df = pd.read_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/State Month Species Mean R2 Val and P Val All Droughts.csv")
#         df.dropna(inplace = True)
#         df = df[(df['Species'] == bird) & ((df['Month'] == 4) | (df['Month'] == 5) | (df['Month'] == 9) | (df['Month'] == 10)) & (df['Drought Level'] == drought)].copy()
#         state_months = len(df.index)
#         df = df[df['P Value'] <= 0.105]
#         #If there are no significant p_values, we don't add this to the list
#         less_than = len(df.index)
#         percent = (less_than / state_months) * 100
#         spring_count = ((df['Month'] == 4) | (df['Month'] == 5)).sum()
#         fall_count = ((df['Month'] == 9) | (df['Month'] == 10)).sum()
#         li.append(bird)
#         li.append(state_months)
#         li.append(less_than)
#         li.append(percent)
#         li.append(spring_count)
#         li.append(fall_count)
#         li.append(drought)
#         print(bird, state_months, less_than, percent, fall_count, spring_count, drought)
    
# df_chart = pd.DataFrame(np.array(li).reshape(-1, 7), columns = ['Species', 'Number State-Months', 'Number State-Months w/ p <= 0.10', 'Percent of State-Months w/ p<=0.10', 'Number of Spring Months', 'Number of Fall Months', 'Drought Level'] )

# print(df_chart)
# df_chart.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/State Month Breakdown.csv")

df = pd.read_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/State Month Breakdown.csv")
df.dropna(inplace = True)



#---------------------------------------------------
#          Linear Regression on State-Year
#---------------------------------------------------


# r2_values = []
# years = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
#           '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
# for year in years:
#     for i in range(1, 13):
#         df_merge = pd.read_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Mean_Merge.csv", low_memory = False, index_col = [0])
#         df_merge['Year'] = df_merge['Year'].astype(str)
#         df = df_merge[(df_merge['Year'] == year) & (df_merge['Month'] == i)].copy()
#         #print(df)
#         df.replace([np.inf, -np.inf], np.nan, inplace = True)
#         df.dropna(inplace = True)
#         drops = ['COMMON NAME', 'MapDate', 'Month']
#         df.drop(columns = drops, inplace = True)
#         columns = ['State', 'Year']
#         df = df.groupby(columns).mean()
#         #print(len(df), year, i)
#         X = df["D1_or_More"].values.reshape(-1, 1)
#         y = df["Observations/Minute"].values.reshape(-1, 1)
#         # print(X)
#         # print(y)
#         lm = linear_model.LinearRegression()
#         lm.fit(X,y)
#         y_pred = lm.predict(X)
        
#         states_for_loop = ['AL', 'GA', 'FL', 'LA', 'MS', 'NC', 'SC', 'VA']
#         no_GA = ['AL', 'FL', 'LA', 'MS', 'NC', 'SC', 'VA']
#         no_LA = ['AL', 'GA', 'FL', 'MS', 'NC', 'SC', 'VA']
#         no_AL = ['GA', 'FL', 'LA', 'MS', 'NC', 'SC', 'VA']
#         # years = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
#         #           '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
#         # years_no_2002 = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
#         #           '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
#         # years_no_2003 =  ['2002', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
#         #           '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
#         lr = LinearRegression()
#         lr.fit(X, y)
#         y_pred = lr.predict(X)
#         plt.scatter(X, y)
#         plt.plot(X, y_pred, color = "red")
#         title = "Linear Regression:", year, ", All Warblers,", i
#         plt.title(title)
#         plt.xlabel("D1_or_More")
#         plt.ylabel("Observations/Minute")
#         annotation = 'R^2 = ', r2_score(y, y_pred)
#         plt.annotate(annotation, xy = (10, 0.01))
#         #2003 7 is missing LA
#         #2002 7 is missing GA
#         #2002 1 is missing AL
#         if year == '2003' and i == 7:
#             for j,txt in enumerate(no_LA):
#                     plt.annotate(txt, (X[j], y[j]))
#         elif year == '2002' and i == 7:
#             for j,txt in enumerate(no_GA):
#                     plt.annotate(txt, (X[j], y[j]))
#         elif year == '2002' and i == 1:
#             for j,txt in enumerate(no_AL):
#                 plt.annotate(txt, (X[j], y[j]))
#         else:
#             for j,txt in enumerate(states_for_loop):
#                     plt.annotate(txt, (X[j], y[j]))
                   
#         plt.show()
        
#         r2_values.append(year)
#         r2_values.append(i)
#         r2_values.append(r2_score(y, y_pred))
#         print(year, i, r2_score(y, y_pred))

# #print(r2_values)
# df_r2 = pd.DataFrame(np.array(r2_values).reshape(-1, 3), columns = ['Year', 'Month', 'R^2 Value'] )
# #print(df_r2)
# df_r2.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/Year Month Mean R2 Values D1 or More.csv")
# # #df.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/All Warblers Linear Reg.csv")


#--------------------------------------------------
#      Linear Regression on State-Month, All Warblers
#--------------------------------------------------


# r2_values = []
# state_list = ['AL', 'GA', 'FL', 'LA', 'MS', 'NC', 'SC', 'VA']
# species = ["Ovenbird","Worm-eating Warbler","Louisiana Waterthrush",
#            "Northern Waterthrush","Golden-winged Warbler","Blue-winged Warbler",
#            "Black-and-white Warbler","Prothonotary Warbler","Swainson's Warbler",
#            "Tennessee Warbler","Orange-crowned Warbler","Nashville Warbler",
#            "Connecticut Warbler","Mourning Warbler","Kentucky Warbler",
#            "Common Yellowthroat","Hooded Warbler","American Redstart","Cape May Warbler",
#            "Cerulean Warbler","Northern Parula","Magnolia Warbler","Bay-breasted Warbler",
#            "Blackburnian Warbler","Yellow Warbler","Chestnut-sided Warbler",
#            "Blackpoll Warbler","Black-throated Blue Warbler","Palm Warbler","Pine Warbler",
#            "Yellow-rumped Warbler","Yellow-throated Warbler","Prairie Warbler",
#            "Black-throated Green Warbler","Canada Warbler","Wilson's Warbler"]
# for state in state_list:
#     for i in range(1, 13):
#         df_merge = pd.read_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Mean_Merge.csv", low_memory = False, index_col = [0])
#         df = df_merge[(df_merge['Month'] == i) & (df_merge['State'] == state)].copy()
#         #print(df)
#         df.replace([np.inf, -np.inf], np.nan, inplace = True)
#         df.dropna(inplace = True)
#         drops = ['COMMON NAME', 'MapDate', 'State']
#         df.drop(columns = drops, inplace = True)
#         columns = ['Year', 'Month']
#         df = df.groupby(columns).mean()
#         #print(df)
#         X = df["D1_or_More"].values.reshape(-1, 1)
#         y = df["Observations/Minute"].values.reshape(-1, 1)
#         # print(X)
#         # print(y)
#         lm = linear_model.LinearRegression()
#         lm.fit(X,y)
#         y_pred = lm.predict(X)
        
#         years = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
#                   '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
#         years_no_2002 = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
#                   '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
#         years_no_2003 =  ['2002', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
#                   '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
#         lr = LinearRegression()
#         lr.fit(X, y)
#         y_pred = lr.predict(X)
#         plt.scatter(X, y)
#         plt.plot(X, y_pred, color = "red")
#         title = "Linear Regression:", state, ", All Warblers,", i
#         plt.title(title)
#         plt.xlabel("D1_or_More")
#         plt.ylabel("Observations/Minute")
#         annotation = 'R^2 = ', r2_score(y, y_pred)
#         plt.annotate(annotation, xy = (60, 0.005))
#         #LA 7 missing 2003
#         #AL 1, GA 7 missing 2002
#         if (state == 'AL' and i == 1) or (state == 'GA' and i == 7):
#             for j,txt in enumerate(years_no_2002):
#                     plt.annotate(txt, (X[j], y[j]))
#         elif state == 'LA' and i == 7:
#             for j,txt in enumerate(years_no_2003):
#                     plt.annotate(txt, (X[j], y[j]))
#         else:
#             for j,txt in enumerate(years):
#                     plt.annotate(txt, (X[j], y[j]))
                
            
#         plt.show()
        
#         r2_values.append(state)
#         r2_values.append(i)
#         r2_values.append(r2_score(y, y_pred))
#         print(state, i, r2_score(y, y_pred))

# #print(r2_values)
# df_r2 = pd.DataFrame(np.array(r2_values).reshape(-1, 3), columns = ['State', 'Month', 'R^2 Value'] )
# #print(df_r2)
# df_r2.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/State Month Mean R2 Values D1 or More.csv")
# #df.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/All Warblers Linear Reg.csv")


#-------------------------------------------------------------
#                  Running Regressions v1
#-------------------------------------------------------------


# state_list = ['AL', 'GA', 'FL', 'LA', 'MS', 'NC', 'SC', 'VA']
# r2_values = []
# for state in state_list:
#     for i in range(1, 13):
#         df_merge = pd.read_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/02_to_19_Mean_Merge.csv", low_memory = False, index_col = [0])
#         df1 = df_merge[(df_merge['State'] == state) & (df_merge['Month'] == i)]
#         df = df1[['Total_Drought_Area', 'Observations/Minute', 'D1_or_More']].copy()
#         # print(df.shape)
#         # print(df)
#         #Some obs/min turned out to be infinity since they were divided by zero, we have to drop those rows unfortunately. Can go back and fix at the source
#         df.replace([np.inf, -np.inf], np.nan, inplace = True)
#         df.dropna(inplace = True)
#         # print(df.shape)
#         X = df["Total_Drought_Area"].values.reshape(-1, 1)
#         y = df["Observations/Minute"].values.reshape(-1, 1)
#         # # Note the difference in argument order
#         # model = sm.OLS(y, X).fit()
#         # predictions = model.predict(X) # make the predictions by the model
        
#         # # Print out the statistics
#         # print(model.summary())
        
#         lm = linear_model.LinearRegression()
#         lm.fit(X,y)
#         y_pred = lm.predict(X)
#         r2_values.append(state)
#         r2_values.append(i)
#         r2_values.append(r2_score(y, y_pred))
#         print(state, i, r2_score(y, y_pred))

# print(r2_values)
# df_r2 = pd.DataFrame(np.array(r2_values).reshape(-1, 3), columns = ['State', 'Month', 'R^2 Value'] )
# print(df_r2)
# df_r2.to_csv("C:/Users/ccsuc/Desktop/Spring 2020 Classes/Senior Research/State Month R2 Values.csv")

