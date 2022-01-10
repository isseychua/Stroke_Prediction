import pandas as pd
import numpy as np

from collections import Counter
from sklearn.preprocessing import LabelEncoder

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

def check_fill_nan(df_all_data,list_nan_col):
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
    print(df_all_data.isnull().sum())
    for col in list_nan_col:
        replacement=df_all_data[col].dropna().mean()
        df_all_data[col]=df_all_data[col].fillna(replacement)
    return df_all_data

def check_drop_duplicated(df_all_data):
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
    print("Number of duplicated rows {}".format(df_all_data.duplicated().sum()))

def check_outliers(df_all_data, list_num_col_less_id):
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
    for feature in list_num_col_less_id:
        lower_limit=df_all_data[feature].quantile(0.25)
        upper_limit=df_all_data[feature].quantile(0.75)
        iqr=upper_limit-lower_limit
        iqr_step=iqr*1.5
        feature_outlier_index=df_all_data[(df_all_data[feature]<lower_limit-iqr_step)|(df_all_data[feature]>upper_limit+iqr_step)].index
        outlier_index.extend(feature_outlier_index)
    
    outlier_index_counter=Counter(outlier_index)

    outlier_index_unique=[]
    for key,value in outlier_index_counter.items():
        outlier_index_unique.append(key)
    return outlier_index_unique

def drop_outliers(df_all_data, outlier_index_unique):
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
    df_all_data=df_all_data.drop(outlier_index_unique,axis=0).reset_index(drop=True)
    return df_all_data

def Label_Encoder(df_all_data,list_object_col):
    encoder=LabelEncoder()
    for col in list_object_col:
        df_all_data[col]=encoder.fit_transform(df_all_data[col])
    return df_all_data

def Filter_columns(self):
    self.df=self.df[["age","avg_glucose_level","stroke"]]
'''
def Filtered_df(self,age_limit,glucose_limit):
    self.df=self.df[(self.df["age"]>age_limit) & (self.df["avg_glucose_level"]<glucose_limit)]
    print(self.df.shape)

def Mean_Std_Calculation(self):
    self.Age_Mean=self.stroke_df["age"].mean()
    self.Age_Std=self.stroke_df["age"].std()
    self.Glucose_Mean=self.stroke_df["avg_glucose_level"].mean()
    self.Glucose_Std=self.stroke_df["avg_glucose_level"].std()
    print(self.Age_Mean)
    print(self.Age_Std)
    print(self.Glucose_Mean)
    print(self.Glucose_Std)
'''
