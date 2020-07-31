# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:26:12 2020

@author: Michael Frangos
"""
#Import number processing library
import numpy as np



#Change working directory
import os
main_directory = 'C:/Users/MF/Desktop/MSF/Project 1'
os.chdir(main_directory)

#load data
import pandas as pd
data = pd.read_excel("project1_data.xlsx")

#Create year column
data["year"] = [date.year for date in data["Date"]]

def create_period_column():
    #Initialize empty [period] column
    data["period"] = 0
    #Create date filters. This will be used to classify each date as period 1 or 2
    period_1 = (data['Date'] > '1974-1-1') & (data['Date'] <= '1988-1-1') #Set up period 1, 1974 - 1988
    period_2 = (data['Date'] > '1988-1-1') & (data['Date'] <= '2020-1-1') #Set up period 2, 1988 - 2020
    print("There is",len(data.loc[period_1]), "entries in period 1")
    print("There is",len(data.loc[period_2]), "entries in period 2")
    #By using the date filters above, replace 0 with 1 in the [period] column for each respective date
    data.loc[data[period_1].index,"period"] = data.loc[data[period_1].index, 'period'].replace(to_replace = 0,value = 1)
    data.loc[data[period_2].index,"period"] = data.loc[data[period_2].index, 'period'].replace(to_replace = 0,value = 2)

#Create the period column and append it to the dataframe
create_period_column()

#Create counter column and append it to the dataframe
data["counter"] = range(len(data))

#####Create Hedged Index#####

#Define a function to create forward premium columns for all of the forward ratios by subtracting 1. Appends the new columns to the main dataframe
def create_forward_premium_columns():
    forward_ratio_list = ['USCAN_fwd', 'USFRN_fwd', 'USGER_fwd', 'USJAP_fwd', 'USUK_fwd']
    for forward_ratio in forward_ratio_list:
        print(f"Successfully created {forward_ratio} premium column")
        data[f"{forward_ratio} premium"] = data[forward_ratio] - 1  
#Execute function to create forward premium columns and append to the dataframe       
create_forward_premium_columns()


#Define function to add to return index to create hedged return index. Appends the new columns to the main dataframe
def create_hedged_currency_returns_columns():
    forward_premium_list = ['USCAN_fwd premium', 'USFRN_fwd premium', 'USGER_fwd premium', 'USJAP_fwd premium', 'USUK_fwd premium']
    unhedged_local_currency_list = ['LCUH_CAN', 'LCUH_FRN', 'LCUH_GER', 'LCUH_JAP', 'LCUH_UK']
    #For each column in the unhedged local currency list and the forward ratio list, add each column together to create the hedged return index columns
    for unhedged_local_currency,forward_premium in zip(unhedged_local_currency_list,forward_premium_list):
        #print(unhedged_local_currency,forward_ratio)
        ##Create the column for US dollar based returns.
        data[f"Hedged_LC_{unhedged_local_currency[-3:]}"] = data[unhedged_local_currency] + data[forward_premium] #forward premium is the hedge. 
        print(f"Successfully created and appended Hedged_LC_{unhedged_local_currency[-3:]} column")
        
#Execute function to create hedged return index. These are the currency returns assuming you hedged
create_hedged_currency_returns_columns()


#Import describe function to generate summary statistics
from scipy.stats import describe
#Define a function to compute summary statistics for each hedged/unhedged currency return column
def create_summary_stats_dataframe(data):
    #We will iterate through these columns
    hedged_currency_return_list = ["Hedged_LC_CAN","Hedged_LC_FRN","Hedged_LC_GER","Hedged_LC_JAP","Hedged_LC__UK"] #These are canadian market returns but in US dollar returns.
    unhedged_market_return_list = ['CANPX', 'FRNPX', 'GERPX', 'JAPPX', 'UKPX','USPX'] #These are canadian market returns but in US dollar returns.
    #Initialize empty dataframe
    dataframe = pd.DataFrame()
    #Initialize empty list to Keep track of row names
    row_name_list = []
    #We will begin by iterating through each column of hedged and unhedged returns
    for hedged_return,unhedged_return in zip(hedged_currency_return_list,unhedged_market_return_list):
        #Compute summary statistics for the hedged return column
        summary = describe(data[f"{hedged_return}"], axis=0)

        #Creates first row of the summary table if it doesn't exist yet
        if len(dataframe) == 0:
            #Create the summary table by initiating the first row.
            dataframe = pd.DataFrame([summary], columns=summary._fields)
            #Generate and append the first unhedged return summary stats row to the summary table
            unhedged_summary = describe(data[f"{unhedged_return}"], axis=0)
            dataframe = dataframe.append(pd.DataFrame([unhedged_summary], columns=unhedged_summary._fields))
        #If first row is already created, created the new rows   
        else:
            #append hedged return summary row to the summary statistics table
            dataframe = dataframe.append(pd.DataFrame([summary], columns=summary._fields))
            print(dataframe)
            
            #append unhedged return summary row to the summary statistics table
            unhedged_summary = describe(data[f"{unhedged_return}"], axis=0)
            dataframe = dataframe.append(pd.DataFrame([unhedged_summary], columns=unhedged_summary._fields))
        #Keep track of each row name. We will use this list to name the rows later
        row_name_list.append(f"{hedged_return}")
        row_name_list.append(f"{unhedged_return}")
    
    #Set the names of each row to the dataframe
    dataframe.index = row_name_list

    return dataframe

#Filter the data by [period] column so we can tabulate results for each [period] column
period_1_data = data.loc[data["period"] == 1]
period_2_data = data.loc[data["period"] == 2]

#Create summary statistics tables
total_sample_summary_stats_dataframe = create_summary_stats_dataframe(data)
period_1_summary_stats_dataframe = create_summary_stats_dataframe(period_1_data)
period_2_summary_stats_dataframe = create_summary_stats_dataframe(period_2_data)

#Export summary statistics tables to excel
total_sample_summary_stats_dataframe.to_excel("total_sample_summary_stats_dataframe.xlsx")
period_1_summary_stats_dataframe.to_excel("period_1_summary_stats_dataframe.xlsx")
period_2_summary_stats_dataframe.to_excel("period_2_summary_stats_dataframe.xlsx")

###Perform t_test###
#import t-test function
from scipy.stats import ttest_rel

#Defines function to perform t-tests on TWO RELATED samples of scores, hedged vs unhedged currency returns, 
#to see if their mean returns are different. 
#Iterates through hedged/unhedged currency return columns and returns t-stat tables.
def perform_t_tests_on_data(data):
    unhedged_market_return_list = ['CANPX', 'FRNPX', 'GERPX', 'JAPPX', 'UKPX']
    hedged_currency_return_list = ["Hedged_LC_CAN","Hedged_LC_FRN","Hedged_LC_GER","Hedged_LC_JAP","Hedged_LC__UK"]
    t_statistic_results_list = []
    p_statistic_results_list = []
    row_name_list = []
    #For each set of columns, Calculate the t-test on TWO RELATED samples of scores, a and b.
    for unhedged_return_array, hedged_return_array in zip(unhedged_market_return_list,hedged_currency_return_list): 
        #Perform t-test
        t, p = ttest_rel(data[unhedged_return_array],data[hedged_return_array])
        #Keep track of the results in lists
        t_statistic_results_list.append(t)
        p_statistic_results_list.append(p)
        row_name_list.append(unhedged_return_array[:-2])
        
    #Compile the results by appending our lists together into a dataframe
    results = pd.DataFrame({"t":t_statistic_results_list,
                            "p":p_statistic_results_list},
                            index = row_name_list)
    return results

#Perform t-tests for hedged/unhedged currency returns during period 1, period 2, and the total sample.
t_test_total_sample_dataframe = perform_t_tests_on_data(data)
t_test_period_1_dataframe = perform_t_tests_on_data(period_1_data)
t_test_period_2_dataframe = perform_t_tests_on_data(period_2_data)

#Export t-test tables to excel
#Perform t-tests
t_test_total_sample_dataframe.to_excel("t_test_total_sample_dataframe.xlsx")
t_test_period_1_dataframe.to_excel("t_test_period_1_dataframe.xlsx")
t_test_period_2_dataframe.to_excel("t_test_period_2_dataframe.xlsx")



#Import libraries for data visualization
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#Create currency correlation tables
def create_correlation_tables():
    #Create a list of column names
    columns = data[["Hedged_LC_CAN","Hedged_LC_FRN","Hedged_LC_GER"
                       ,"Hedged_LC_JAP","Hedged_LC__UK",'CANPX', 
                       'FRNPX', 'GERPX', 'JAPPX', 'UKPX']].columns
    
    #Create correlation table
    total_sample_corr_table = data[columns].corr()
    #Create correlation table
    period_1_corr_table = period_1_data[columns].corr()
    #Create correlation table
    period_2_corr_table = period_2_data[columns].corr()
    
    return total_sample_corr_table, period_1_corr_table, period_2_corr_table

#Create currency correlation tables
total_sample_corr_table, period_1_corr_table, period_2_corr_table = create_correlation_tables()

#Visualize and export hedged and unhedged currency correlation tables
def visualize_currency_correlation_tables(): 
    #Create a list of column names
    columns = data[["Hedged_LC_CAN","Hedged_LC_FRN","Hedged_LC_GER"
                       ,"Hedged_LC_JAP","Hedged_LC__UK",'CANPX', 
                       'FRNPX', 'GERPX', 'JAPPX', 'UKPX']].columns                        
    #Place the column names into a list to make the correlation table visualization.            
    labels = [x for x in columns]

    #Creates lists for that contain the numerical tables and their names
    tables = [total_sample_corr_table, period_1_corr_table, period_2_corr_table]
    table_names = ["Total Sample Correlations", "Period 1 Correlations", "Period 2 Correlations"]

    for table,table_name in zip(tables,table_names):
        #Create empty figure objects that contain visuals. We will populate them with the next few lines of code
        fig = plt.figure(figsize = (12,12))
        ax = fig.add_subplot(111)
        #Select the table used to create the visualization, and populate the figure object
        ax.matshow(table,cmap = plt.cm.RdYlGn,alpha=0.4)
        #Set the names of the x ticks & y ticks
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        #Set title
        plt.title(f"{table_name}", y=1.14)
        #Force show all of the labels since they are being truncated by default
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        
        #Rotate the xticks vertically
        ax.set_xticklabels(labels,rotation=90)    
        
        #Visualize the numbers onto the correlation table
        numbers_to_vizualize = np.array(table)
        for (i, j), z in np.ndenumerate(numbers_to_vizualize):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        
        #Export numerical tables
        table.to_excel("Total_Sample_Correlations.xlsx")
        #Export Table visualizations
        fig.savefig(f"{table_names}.png")

#Visualize and export hedged and unhedged currency correlation tables
visualize_currency_correlation_tables()

'''
#Create the independent variables for the linear regression.
'''

#Import libraries for regression
import statsmodels.api as models
def perform_regressions(dataset, data_set_name):
    #Set up lists to iterate through to generate our new variables
    independent_var_name_list = ["x_can","x_frn","x_ger","x_jap","x_uk"]
    unhedged_market_return_list = ['CANPX', 'FRNPX', 'GERPX', 'JAPPX', 'UKPX']
    currency_return_list = ['CUR_RET_CAN', 'CUR_RET_FRN', 'CUR_RET_GER', 'CUR_RET_JAP','CUR_RET_UK']
    forward_premium_list = ['USCAN_fwd premium', 'USFRN_fwd premium', 'USGER_fwd premium', 'USJAP_fwd premium', 'USUK_fwd premium']
    #Initialize empty lists to populate with standard errors and OHR by currency
    standard_error_list = []
    Optimal_Hedge_ratio_list = []
    #Iterate through all three lists at the same time
    for independent_var,unhedged_return,currency_return, forward_premium in zip(independent_var_name_list,unhedged_market_return_list,currency_return_list,forward_premium_list):
        #Print new line, name and variable
        print(dataset_name, f" - {independent_var}")
        #The dependent variables are unhedged returns
        #Create the independent variable columns and append to the main dataframe
        dataset[f"{independent_var}"] = dataset[currency_return] - dataset[forward_premium]
        #Set up for regression
        #x = data[[f"{independent_var}"]]
        x = dataset[[f"{independent_var}"]]
        y = dataset[[f"{unhedged_return}"]]
        
        # Fit & print regression model
        regr = models.OLS(y, x).fit()
        print_model = regr.summary()
        print(print_model)
        
        standard_error_list.append(regr.bse)
        Optimal_Hedge_ratio_list.append(regr.params)
        
    return standard_error_list, Optimal_Hedge_ratio_list
        

#Update date filters
#Filter the data by [period] column so we can tabulate results for each [period] column
period_1_data = data.loc[data["period"] == 1]
period_2_data = data.loc[data["period"] == 2]

#Execute regressions for all currencies and all data sets. #Beta is the optimal hedge ratio
list_of_datasets = [data, period_1_data, period_2_data]
list_of_dataset_names = ["Entire Sample", "Period 1", "Period 2"]
for dataset, dataset_name in zip(list_of_datasets,list_of_dataset_names):
    perform_regressions(dataset,dataset_name)
    










#Perform regression over 36 months and sliding by 1 increment
t = 0 #Time
window = 36
Trailing_36m_SE_row = []  #We will populate this empty list with each iteration's results by currency. Ex. [SE_can,SE_UK ....]
Trailing_36m_OHR_row = [] #We will populate this empty list with each iteration's results by currency. 
#Iterate through the dataset using 36 month windows to calculate rolling regressions
for t in range(len(data)-36):
    print(t)
    #Slice and roll through the dataset by the window index
    sliced_dataset = data[t:t+window]
    print(sliced_dataset)
    #Execute regressions for all currencies. #Beta is the optimal hedge ratio
    standard_error, Optimal_Hedge_Ratio = perform_regressions(sliced_dataset,"Entire Sample")
    #Keep track of the results. We will append these new columns to our dataframe at the end.
    Trailing_36m_SE_row.append(standard_error)
    Trailing_36m_OHR_row.append(Optimal_Hedge_Ratio)

#Now that the rolling regressions have been calculated, let's append them to our dataframe
def append_rolling_regression_SE_columns_to_dataframe():
    #Initialize empty lists and generate NANs by the size of the window
    CAN_36M_SE, FRN_36M_SE, GER_36M_SE, JAP_36M_SE, UK_36M_SE = [np.nan for x in range(window)],[np.nan for x in range(window)],[np.nan for x in range(window)],[np.nan for x in range(window)],[np.nan for x in range(window)]
    
    #Populate the empty lists with the calculated data
    i=0
    for CAN, FRN, GER, JAP, UK in Trailing_36m_SE_row:
        print("\n",CAN, FRN, GER, JAP, UK,i)
        i=i+1
        CAN_36M_SE.append(float(CAN))
        FRN_36M_SE.append(float(FRN))
        GER_36M_SE.append(float(GER))
        JAP_36M_SE.append(float(JAP))
        UK_36M_SE.append( float(UK) )
    #Append columns to the main dataframe
    data["CAN_36M_SE"] = CAN_36M_SE
    data["FRN_36M_SE"] = FRN_36M_SE
    data["GER_36M_SE"] = GER_36M_SE
    data["JAP_36M_SE"] = JAP_36M_SE
    data["UK_36M_SE"] = UK_36M_SE
    
#Now that the rolling regressions have been calculated, let's append them to our dataframe
def append_rolling_regression_coefficient_columns_to_dataframe():
    #Initialize empty lists
    CAN_36M_OHR, FRN_36M_OHR, GER_36M_OHR, JAP_36M_OHR, UK_36M_OHR = [np.nan for x in range(window)],[np.nan for x in range(window)],[np.nan for x in range(window)],[np.nan for x in range(window)],[np.nan for x in range(window)]
    #Populate the empty lists
    for CAN, FRN, GER, JAP, UK in Trailing_36m_OHR_row:
        CAN_36M_OHR.append(float(CAN))
        FRN_36M_OHR.append(float(FRN))
        GER_36M_OHR.append(float(GER))
        JAP_36M_OHR.append(float(JAP))
        UK_36M_OHR.append(float(UK))
    #Append columns to the main dataframe
    data["CAN_36M_OHR"] = CAN_36M_OHR
    data["FRN_36M_OHR"] = FRN_36M_OHR
    data["GER_36M_OHR"] = GER_36M_OHR
    data["JAP_36M_OHR"] = JAP_36M_OHR
    data["UK_36M_OHR"] = UK_36M_OHR
            
#Execute functions: Append the new columns to the dataframe
append_rolling_regression_SE_columns_to_dataframe()
append_rolling_regression_coefficient_columns_to_dataframe()

#Replace values in the rolling regressions columns to prevent overhedging. Then print histograms
_36M_OHR_list = ["CAN_36M_OHR","FRN_36M_OHR","GER_36M_OHR","JAP_36M_OHR","UK_36M_OHR"]
for column in _36M_OHR_list : 
    #Replace values less than zero with zero
    data[f"{column}"].loc[(data[f"{column}"] < 0 )] = 0
    #Replace values greater than one with one
    data[f"{column}"].loc[(data[f"{column}"] > 1 )] = 1
    #Print histograms
    plt.hist(data[f"{column}"])
    plt.title(f"{column}")
    plt.show()
    

#Tabstat Optimal Hedge Ratios
subdir_1 = "OHR Regression Tabstats"
Total_Period = pd.DataFrame(data[_36M_OHR_list].describe().T)
a = data[["CAN_36M_OHR","period"]].groupby("period").describe()
b = data[["FRN_36M_OHR","period"]].groupby("period").describe()
c = data[["GER_36M_OHR","period"]].groupby("period").describe()
d = data[["JAP_36M_OHR","period"]].groupby("period").describe()
e = data[["UK_36M_OHR","period"]].groupby("period").describe()
#Export OHR tabstats to excel
File_names = ['Total_Sample','CAN_36M_OHR','FRN_36M_OHR','GER_36M_OHR','JAP_36M_OHR','UK_36M_OHR']
OHR_Reg_tabstats = [Total_Period,a,b,c,d,e]
for file, filename in zip (OHR_Reg_tabstats,File_names):
    file.to_excel(f"{main_directory}/{subdir_1}/{filename}.xlsx")




#Calculate confidence intervals for the OHR rolling Regressions
def calculate_OMH_regression_confidence_intervals():
    a_group = data[["CAN_36M_OHR","period"]].groupby("period")
    b_group = data[["FRN_36M_OHR","period"]].groupby("period")
    c_group = data[["GER_36M_OHR","period"]].groupby("period")
    d_group = data[["JAP_36M_OHR","period"]].groupby("period")
    e_group = data[["UK_36M_OHR","period"]].groupby("period")
    
    #Print confidence intervals for periods
    groups = [a_group,b_group,c_group,d_group,e_group]
    group_names = ['CAN_36M_OHR','FRN_36M_OHR','GER_36M_OHR','JAP_36M_OHR','UK_36M_OHR']
    from scipy.stats import norm
    for group,group_name in zip(groups,group_names):
        mu = group.mean()
        standard_error = group.std()/np.sqrt(group.count())
        confidence_interval = lower, upper = norm.interval(0.95, loc=mu, scale=standard_error)
        print(group_name, "Lower Confidence Intervals:", list(lower), "Upper Confidence Intervals:", upper)
    
    #Print confidence intervals for total sample for each country
    Total_Sample_group = pd.DataFrame(data[_36M_OHR_list])
    mu = Total_Sample_group.mean()
    standard_error = Total_Sample_group.std()/np.sqrt(Total_Sample_group.count())
    confidence_interval = lower, upper = norm.interval(0.95, loc=mu, scale=standard_error)
    print("Total_Sample_Group", "Lower Confidence Intervals:", list(lower), "Upper Confidence Intervals:", upper)
    
#Execute function
calculate_OMH_regression_confidence_intervals()


#Print histograms by currency & period group
_36M_OHR_list = ["CAN_36M_OHR","FRN_36M_OHR","GER_36M_OHR","JAP_36M_OHR","UK_36M_OHR"]
for column in _36M_OHR_list: 
    ###By period - Red
    data[f"{column}"].hist(by=data['period'],alpha=1)
    plt.title(f"{column} - By Period \n 2")
    
    ###Total sample - Blue
    data[f"{column}"].hist(alpha=.50)
    plt.legend(["By Period", f"Total Sample - {column}"]) 
    plt.show()



#Japan OHR plot
plt.plot(data["Date"],data["JAP_36M_OHR"])
plt.title("JAP 36M OHR")

Ending_dataset = data.to_excel("Ending_dataset.xlsx")
