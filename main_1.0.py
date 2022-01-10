#Import packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from packages.project_mod import preprocessing, data_categorisation, eda, featureengineering, modelling

#Retrieve data from dataset and inspect dataset
df_all_data=preprocessing("./Source/healthcare-dataset-stroke-data.csv")
#Break the columns into list of numerical, categorical and object type.
list_all_col, list_num_col, list_cat_col, list_object_col, list_algo_model_col, list_num_col_less_id=data_categorisation(df_all_data)
#Inspect univariate and bivariate models
eda(df_all_data,list_num_col,list_cat_col)
#Impute nan, drop duplicate values, drop outlier rows and label encode dataset.
df_all_data=featureengineering(df_all_data,list_num_col_less_id,list_object_col)
#Due to the imbalance dataset, we will first train test split the data using StratifiedKfold, oversample using RandomOverSample and finally predict the outcome. Final results will be guage using f1 score and exported into an CSV file.
modelling(df_all_data,list_algo_model_col)

