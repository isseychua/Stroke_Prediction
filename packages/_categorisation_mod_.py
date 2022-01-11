#Import packages
import pandas as pd
import numpy as np

def list_of_columns(df_all_data):
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
    list_all_col=df_all_data.columns.tolist()
    list_num_col=["id","age","avg_glucose_level","bmi"]
    list_num_col_less_id=["age","avg_glucose_level","bmi"]
    list_cat_col=["hypertension","heart_disease","gender","ever_married","work_type","Residence_type","smoking_status","stroke"]
    list_object_col=["gender","ever_married","work_type","Residence_type","smoking_status"]
    list_algo_model_col=["Lr","Bagging","Boosting","Stacking"]
    return list_all_col, list_num_col, list_cat_col, list_object_col, list_algo_model_col, list_num_col_less_id
