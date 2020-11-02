import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

import statsmodels.api as sm
from statsmodels.tsa.api import Holt

import warnings
warnings.filterwarnings("ignore")

######## time series #######
def split_data(df):
    '''splits into train, validate, test'''
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]
    # print the shape of each df
    train.shape, validate.shape, test.shape
    return train, validate, test

def sanity_check_split(df1, train, validate, test):
    '''checks train, validate, test splits'''
    # Does the length of each df equate to the length of the original df?
    print('df lengths add to total:', len(train) + len(validate) + len(test) == len(df1))
    # Does the first row of original df equate to the first row of train?
    print('1st row of full df == 1st row train:', df1.head(1) == train.head(1))
    # Is the last row of train the day before the first row of validate? And the same for validate to test?
    print('\n Is the last row of train the day before the first row of validate? And the same for validate to test?')
    print(pd.concat([train.tail(1), validate.head(1)]))
    print(pd.concat([validate.tail(1), test.head(1)]))
    # Is the last row of test the same as the last row of our original dataframe?
    print('\n Is the last row of test the same as the last row of our original dataframe?')
    print(pd.concat([test.tail(1), df1.tail(1)]))

def chart_splits(train, validate, test):
    for col in train.columns:
        plt.plot(train[col])
        plt.plot(validate[col])
        plt.plot(test[col])
        plt.ylabel(col)
        plt.title(col)
        plt.show()

def evaluate(target_var):
    '''evaluate() will compute the Mean Squared Error and the Rood Mean Squared Error to evaluate'''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse

def plot_and_eval(target_var):
    '''
    plot_and_eval() will use the evaluate function and also plot train and test values with the predicted
     values in order to compare performance.
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

def create_eval_df():
    # Create empty dataframe to store model results for comparison
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    return eval_df

# function to store the rmse so that we can compare, note: need to run create_eval_df before this function
def append_eval_df(model_type, target_var):
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)