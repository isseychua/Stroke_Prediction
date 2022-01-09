import pandas as pd
import numpy as np

from collections import Counter

class Preprocessing_Class():

    #Read csv file into dataframe
    def __init__(self,directory_path) -> None:
        self.df=pd.read_csv(directory_path)
    
    def inspect_dataframe(self):
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
        print(self.df.head())
        print(self.df.info())
        print(self.df.describe())
    
    def inspect_stroke(self):
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
        print(self.df.stroke.value_counts(normalize=True))
    
    def list_of_columns(self):
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
        self.Col=self.df.columns.tolist()
        self.Num_Col=["id","age","avg_glucose_level","bmi"]
        self.Cat_Col=["hypertension","heart_disease","gender","ever_married","work_type","Residence_type","smoking_status","stroke"]
        self.Type_Obj=["gender","ever_married","work_type","Residence_type","smoking_status"]
        self.Result_Table_Column=["Lr","Bagging","Boosting","Stacking"]


    def check_fill_nan(self,List_of_nan):
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
        print(self.df.isnull().sum())
        for col in List_of_nan:
            replacement=self.df[col].dropna().mean()
            self.df[col]=self.df[col].fillna(replacement)

    def check_drop_duplicated(self):
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
        print(self.df.duplicated().sum())

    def drop_id_col(self):
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
        self.Num_Col_less_id=self.Num_Col
        self.Num_Col_less_id.remove("id")

    def check_outliers(self):
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
        for feature in self.Num_Col_less_id:
            lower_limit=self.df[feature].quantile(0.25)
            upper_limit=self.df[feature].quantile(0.75)
            iqr=upper_limit-lower_limit
            iqr_step=iqr*1.5
            feature_outlier_index=self.df[(self.df[feature]<lower_limit-iqr_step)|(self.df[feature]>upper_limit+iqr_step)].index
            outlier_index.extend(feature_outlier_index)
        
        outlier_index_counter=Counter(outlier_index)

        self.outlier_index_unique=[]
        for key,value in outlier_index_counter.items():
            self.outlier_index_unique.append(key)

    def drop_outliers(self):
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
        self.df=self.df.drop(self.outlier_index_unique,axis=0).reset_index(drop=True)

    def Export_Results_Csv(self,Filename,index):
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
        self.Result_Table.to_csv(Filename,index=index)