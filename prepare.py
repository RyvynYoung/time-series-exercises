import pandas as pd
import numpy as np
import scipy as sp 
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt

######## time series prepare ######
def prep_store_data(df):
    df.sale_date = pd.to_datetime(df.sale_date, format='%a, %d %b %Y %H:%M:%S %Z')
    df.sale_amount.hist()
    plt.show()
    df.item_price.hist()
    plt.show()
    df = df.set_index('sale_date').sort_index()
    df['month'] = df.index.month
    df['day_week'] = df.index.day_name()
    df['sales_total'] = df.sale_amount * df.item_price
    return df


def prep_germany_data(germany):
    '''prepare dataset for explore'''
    # convert date to datetime for time series analysis
    germany.Date = pd.to_datetime(germany.Date)
    # fill null values with 0
    germany = germany.fillna(0)
    # rename column to python ok type
    germany = germany.rename(columns={'Wind+Solar': 'orig_windsolar'})
    # add calculated wind + solar column
    germany['calc_windsolar'] = germany.Wind + germany.Solar
    # set date as index for time series analysis
    germany = germany.set_index('Date').sort_index()
    
    # create histograms
    for col in germany.columns:
        germany[col].hist()
        plt.title(col)
        plt.show()
    return germany









######## #Clustering Exercises functions ##########
##### Zillow Clustering ########
def remove_columns(df, cols_to_remove):  
    '''
    Remove columns passed to function
    '''
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .5):
    '''
    Drops rows or columns based on the percent of values that are missing, prop_required_column = a number between 0 and 1
    that represents the proportion, for each column, of rows with non-missing values required to keep the column. 
    i.e. if prop_required_column = .6, then you are requiring a column to have at least 60% of values not-NA (no more than 40% missing).
    prop_required_row = a number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing
     values required to keep the row. For example, if prop_required_row = .75, then you are requiring a row to have at least 75% of
    variables with a non-missing value (no more that 25% missing)
    '''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# note: Anthony has a different approach to handle missing values


def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.5):
    '''
    Prep data by removing specificed columns as well as columns as rows with designated proportion of missing values.
    Remember: if prop_required_row = .75, then you are requiring a row to have at least 75% of
    variables with a non-missing value (no more that 25% missing)
    '''
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

def get_counties(df):
    # create dummy vars of fips id
    county_df = pd.get_dummies(df.fips)
    # rename columns by actual county name
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df_dummies = pd.concat([df, county_df], axis = 1)
    return df_dummies

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    outlier_cols = {col + '_up_outliers': get_upper_outliers(df[col], k) for col in df.select_dtypes('number')}
    return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_up_outliers'] = get_upper_outliers(df[col], k)

    return df

def get_lower_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the lower outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the lower bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: max([x - lower_bound, 0]))

def add_lower_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    outlier_cols = {col + '_low_outliers': get_lower_outliers(df[col], k) for col in df.select_dtypes('number')}
    return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_low_outliers'] = get_lower_outliers(df[col], k)
    
    return df

####### example to call outliers functions ####
# malldf = prepare.add_upper_outlier_columns(df, k=1.5)  

# outlier_cols = [col for col in malldf if col.endswith('_outliers')]
# for col in outlier_cols:
#     print('~~~\n' + col)
#     data = malldf[col][malldf[col] > 0]
#     print(data.describe())



def run():
    print("Prepare: Cleaning acquired data...")
    # Write code here
    print("Prepare: Completed!")


