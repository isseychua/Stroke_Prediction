import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE , RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier , AdaBoostClassifier , StackingClassifier
from sklearn.model_selection import GridSearchCV
def Initiate_ML_Algo():
    #Define LogisticRegression, Support Vecotr Classifier, Bagging(RandomForesetClassifier), Boosting, Stacking(SVC,DTC,NB,LR)
    Lr_Model=LogisticRegression(solver="liblinear")
    Lr_Parameters={"penalty":("l1","l2","elasticnet","none"),"dual":[True,False]}
    Gnb_Model=GaussianNB()
    Gnb_Parameters={"var_smoothing":[1e-9,1e-8,1e-7]}
    Dtc_Model=DecisionTreeClassifier()
    Dtc_parameters={"criterion":("gini","entropy"),"splitter":("best","random")}
    Bagging_Model=BaggingClassifier(n_estimators=10,random_state=0)
    Boosting_Model=AdaBoostClassifier(n_estimators=100,random_state=0)
    stacking_estimator=[("Lr",Lr_Model),("Dtc",Dtc_Model),("Gnb",Gnb_Model)]
    Stacking_Model=StackingClassifier(estimators=stacking_estimator,final_estimator=Lr_Model)

    Lr_Grid=GridSearchCV(Lr_Model,Lr_Parameters)
    Gnb_Grid=GridSearchCV(Gnb_Model,Gnb_Parameters)
    return Lr_Grid,Bagging_Model,Boosting_Model,Stacking_Model
def create_result_table(list_algo_model_col):
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
    df_result_table=pd.DataFrame(columns=list_algo_model_col)
    return df_result_table

def x_y_df(df_all_data,Target):
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
    df_x=df_all_data.drop(Target,axis=1)
    df_y=df_all_data[Target]
    return df_x, df_y

def SKFold(n_splits,shuffle,random_state):
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
    KFold=StratifiedKFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)
    return KFold

def algo_calculation(train_X_os , train_Y_os,List_Model,List_Col,test_X,test_Y,df_result_table):
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
    for model in List_Model:
        model.fit(train_X_os,train_Y_os)
        test_Pred=model.predict(test_X)
        Score=f1_score(test_Y,test_Pred)
        Score_List.append(Score)
    df_New_Row=pd.DataFrame([Score_List],columns=List_Col)
    df_result_table=df_result_table.append(df_New_Row,ignore_index=True)
    return df_result_table
def Ave_Score(df_result_table,list_algo_model_col):
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
    for col in list_algo_model_col:
        Score=df_result_table[col].mean()
        Score_List.append(Score)
    df_Ave_Score=pd.DataFrame([Score_List],columns=list_algo_model_col)
    df_result_table=df_result_table.append(df_Ave_Score,ignore_index=True)
    return df_result_table
def KFold_For_Loop(kfold_split,df_x,df_y,list_models,list_algo_model_col,df_result_table):
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
    for train_index, test_index in kfold_split.split(df_x,df_y):
        train_X , test_X=df_x.iloc[train_index],df_x.iloc[test_index]
        train_Y , test_Y=df_y.iloc[train_index],df_y.iloc[test_index]
        train_X_os , train_Y_os=ROS(train_X,train_Y)
        df_result_table=algo_calculation(train_X_os , train_Y_os,list_models,list_algo_model_col,test_X,test_Y,df_result_table)
    return df_result_table
def ROS(train_X,train_Y):
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
    ros_model=RandomOverSampler(random_state=0)
    train_X_os , train_Y_os=ros_model.fit_resample(train_X,train_Y)
    return train_X_os , train_Y_os