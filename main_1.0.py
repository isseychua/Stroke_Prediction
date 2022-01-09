#Import packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from packages.preprocessing_mod import Preprocessing_Class
from packages.eda_mod import eda_class
from packages.feature_engineering_mod import featureengineering_class
from packages.modelling_mod import modelling_class

class main(Preprocessing_Class,eda_class,featureengineering_class,modelling_class):
    pass

#Preprocessing
Main=main("./Source/healthcare-dataset-stroke-data.csv")
#Inspect dataframe
Main.inspect_dataframe()
#Inspect Stroke ratio
Main.inspect_stroke()
#Split into categorical and numerical columns
Main.list_of_columns()
#Check and fill nan
Main.check_fill_nan(["bmi"])
#Check and drop duplicated values
Main.check_drop_duplicated()
#Remove "id" from Num_Col
Main.drop_id_col()
#Check and drop outliers
Main.check_outliers()
Main.drop_outliers()
print(Main.df.shape)

#EDA
#Create dataframe for positive stroke
Main.Create_positive_stroke_df("stroke")
#Uni Variate analysis
Main.univariate_num(2,4)
Main.univariate_cat(2,8)
#Bi Variate analysis
Main.bivariate_num(2,4)
Main.bivariate_cat(2,8)

#Feature Engineering
'''
#Calculate the mean and std of age and avg_glucose_level
Main.Mean_Std_Calculation()
#Filter age>36, avg_glucose_level<90
Main.Filtered_df(Main.Age_Mean-Main.Age_Std*0.5,Main.Glucose_Mean+Main.Glucose_Std*0.5)
'''
#Keep Age and BMI feature
Main.Filter_columns()
'''
#Label Encoder
Main.Label_Encoder(Main.Type_Obj)
'''
#Modelling
Main.Initiate_ML_Algo()
Main.Create_Result_Table(Main.Result_Table_Column)
#Create train and test set
Main.X_Y_df("stroke")
#Train_Test_Split followed by SMOTE oversampling
Main.SKFold(5,True,1)
Main.KFold_For_Loop(Main.Result_Table_Column)
#Calculate the average f1 score for each method
Main.Ave_Score(Main.Result_Table_Column)
print(Main.Result_Table)

#Export table of results in csv
Main.Export_Results_Csv("results_filtered_age_glucose_ros.txt",False)

