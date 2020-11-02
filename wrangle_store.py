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
import acquire
import prepare
import explore


def split_data(df):
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]

    train.shape, validate.shape, test.shape
    print(len(train) + len(validate) + len(test) == len(df))
    return train, validate, test

def wrangle_store_data():
    # get data
    df = acquire.get_store_data()
    # prep data
    df1 = prepare.prep_store_data(df)
    df = df[df.index != '2016-02-29']
    df = df[['sale_amount', 'sales_total']]
    df.resample('D').sum()
    # split and return train, validate, test
    train, validate, test = split_data(df1)
    return train, validate, test



