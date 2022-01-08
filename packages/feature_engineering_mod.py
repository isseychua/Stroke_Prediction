import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

class featureengineering_class():
    def Label_Encoder(self,List_Of_Objects):
        encoder=LabelEncoder()
        for col in List_Of_Objects:
            self.df[col]=encoder.fit_transform(self.df[col])
            