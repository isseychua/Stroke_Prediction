import pandas as pd
import numpy as np

#Read csv file into dataframe
def read_csv_to_df(string_data_dir_path):
    df_all_data=pd.read_csv(string_data_dir_path)
    return df_all_data

def inspect_dataframe(df_all_data):
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
    print(df_all_data.head())
    print(df_all_data.info())
    print(df_all_data.describe())

def inspect_stroke(df_all_data):
    """
    Inspect target stroke for the dataframe. Find the ratio of positive and negative stroke.

    Parameters
    ----------
    df : dataframe
        dataframe to be inspected
    
    Returns
    -------
    None
    """
    print(df_all_data.stroke.value_counts(normalize=True))

def Export_Results_Csv(df_result_table,Filename,index):
    """
    Calculate average f1 score for each model

    Parameters
    ----------
    df : dataframe
        dataframe of models and f1 score for each cross validation
    List : list
        list of models
    
    Returns
    -------
    None
    """
    df_result_table.to_csv(Filename,index=index)