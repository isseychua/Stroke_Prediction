import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

class featureengineering_class():
    def Label_Encoder(self,List_Of_Objects):
        encoder=LabelEncoder()
        for col in List_Of_Objects:
            self.df[col]=encoder.fit_transform(self.df[col])

    def Create_positive_stroke_df(self,Target):
        self.stroke_df=self.df[self.df[Target]==1]
    
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
