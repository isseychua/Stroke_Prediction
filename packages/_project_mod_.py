#Import packages
import pandas as pd
import numpy as np

from packages._preprocessing_mod_ import read_csv_to_df, inspect_dataframe, inspect_stroke, Export_Results_Csv
from packages._eda_mod_ import univariate_num, univariate_cat
from packages._categorisation_mod_ import list_of_columns
from packages._feature_engineering_mod_ import check_fill_nan, check_drop_duplicated, check_outliers, drop_outliers, Label_Encoder
from packages._modelling_mod_ import Initiate_ML_Algo, create_result_table, x_y_df, SKFold, KFold_For_Loop, Ave_Score

def preprocessing(string_data_dir_path):
    #Preprocessing
    df_all_data=read_csv_to_df(string_data_dir_path)
    #Inspect dataframe
    #Inspect Stroke ratio
    inspect_stroke(df_all_data)
    return df_all_data

def data_categorisation(df_all_data):
    #Split into numerical, categorical, object type and algo model, numerical less id columns
    list_all_col, list_num_col, list_cat_col, list_object_col, list_algo_model_col, list_num_col_less_id=list_of_columns(df_all_data)
    return list_all_col, list_num_col, list_cat_col, list_object_col, list_algo_model_col, list_num_col_less_id
    
def eda(df_all_data,list_num_col,list_cat_col):
    #EDA
    #Uni Variate analysis
    univariate_num(df_all_data,list_num_col,2,4)
    univariate_cat(df_all_data,list_cat_col,2,8)
    '''#Bi Variate analysis
    Main.bivariate_num(2,4)
    Main.bivariate_cat(2,8)'''

def featureengineering(df_all_data,list_num_col_less_id,list_object_col):
    #Check and fill nan
    df_all_data=check_fill_nan(df_all_data,["bmi"])
    #Check and drop duplicated values
    check_drop_duplicated(df_all_data)
    #Check and return index row of outlier values
    outlier_index_unique=check_outliers(df_all_data,list_num_col_less_id)
    #Drop rows with outliers
    df_all_data=drop_outliers(df_all_data, outlier_index_unique)
    print(df_all_data.shape)
    #Label Encoder
    df_all_data=Label_Encoder(df_all_data,list_object_col)
    print(df_all_data)
    return df_all_data

def modelling(df_all_data,list_algo_model_col):   
    #Modelling
    Lr_Grid,Bagging_Model,Boosting_Model,Stacking_Model=Initiate_ML_Algo()
    list_models=[Lr_Grid,Bagging_Model,Boosting_Model,Stacking_Model]
    df_result_table=create_result_table(list_algo_model_col)
    #Create train and test set
    df_x, df_y=x_y_df(df_all_data,"stroke")
    #Train_Test_Split followed by SMOTE oversampling
    kfold_split=SKFold(5,True,1)
    df_result_table=KFold_For_Loop(kfold_split,df_x,df_y,list_models,list_algo_model_col,df_result_table)
    #Calculate the average f1 score for each method
    df_result_table=Ave_Score(df_result_table,list_algo_model_col)
    print(df_result_table)
    #Export table of results in csv
    Export_Results_Csv(df_result_table,"results_filtered_age_glucose_ros.txt",False)