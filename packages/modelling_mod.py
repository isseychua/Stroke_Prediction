import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
def algo_calculation(List,Train_X,Train_Y,Test_X,Test_Y):
    """
    Calculate f1 score for each model

    Parameters
    ----------
    List : list
        list of algo models
    Train_X : dataframe
        dataframe of training data with features
    Train_Y : dataframe
        dataframe of training data with target
    Test_X : dataframe
        dataframe of test data with features
    Test_Y : dataframe
        dataframe of test data with target
    
    Returns
    -------
    None
    """
    Score_List=[]
    for model in List:
        model.fit(Train_X,Train_Y)
        Test_Pred=model.predict(Test_X)
        Score=f1_score(Test_Y,Test_Pred)
        Score_List.append(Score)
    return Score_List

def Ave_Score(df,List):
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
    Score_List=[]
    for model in List:
        Score=df[model].mean()
        Score_List.append(Score)
    return Score_List