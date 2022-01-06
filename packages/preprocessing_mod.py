import pandas as pd
import numpy as np

from collections import Counter

def inspect_dataframe(df):
    """
    Inspect head, info and describe for the dataframe

    Parameters
    ----------
    df : dataframe
        dataframe to be inspected
    
    Returns
    -------
    None
    """
    print(df.head())
    print(df.info())
    print(df.describe())

def fill_nan(df,list):
    """
    Fill nan values with either mean or mode

    Parameters
    ----------
    df : dataframe
        dataframe used for checking and filling nan values
    list: list
        list of columns to be checked

    Returns
    -------
    df : dataframe
        modified dataframe with no nan values
    """
    for col in list:
        replacement=df[col].dropna().mean()
        df[col]=df[col].fillna(replacement)
    return df

def check_outliers(df,list):
    """
    Check for upper and lower limit 

    Parameters
    ----------
    df : dataframe
        dataframe to be inspected
    list : list
        list of columns less id
    
    Returns
    -------
    outlier_index_unique : list
        index of rows with outlier values
    """
    outlier_index=[]
    for feature in list:
        lower_limit=df[feature].quantile(0.25)
        upper_limit=df[feature].quantile(0.75)
        iqr=upper_limit-lower_limit
        iqr_step=iqr*1.5
        feature_outlier_index=df[(df[feature]<lower_limit-iqr_step)|(df[feature]>upper_limit+iqr_step)].index
        outlier_index.extend(feature_outlier_index)
    
    outlier_index_counter=Counter(outlier_index)

    outlier_index_unique=[]
    for key,value in outlier_index_counter.items():
        outlier_index_unique.append(key)
    return outlier_index_unique