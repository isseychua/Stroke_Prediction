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

    def Filtered_df(self,age_limit,glucose_limit):
        self.df=self.df[(self.df["age"]>age_limit) & (self.df["avg_glucose_level"]<glucose_limit)]
        print(self.df.shape)