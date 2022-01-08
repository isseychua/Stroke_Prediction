import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold ,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier , AdaBoostClassifier , StackingClassifier
from imblearn.over_sampling import SMOTE , RandomOverSampler
class modelling_class():
    def Initiate_ML_Algo(self):
        #Define LogisticRegression, Support Vecotr Classifier, Bagging(RandomForesetClassifier), Boosting, Stacking(SVC,DTC,NB,LR)
        self.Lr_Model=LogisticRegression(solver="liblinear")
        self.Lr_Parameters={"penalty":("l1","l2","elasticnet","none"),"dual":[True,False]}
        self.Gnb_Model=GaussianNB()
        self.Gnb_Parameters={"var_smoothing":[1e-9,1e-8,1e-7]}
        self.Dtc_Model=DecisionTreeClassifier()
        self.Dtc_parameters={"criterion":("gini","entropy"),"splitter":("best","random")}
        self.Bagging_Model=BaggingClassifier(n_estimators=10,random_state=0)
        self.Boosting_Model=AdaBoostClassifier(n_estimators=100,random_state=0)
        self.stacking_estimator=[("Lr",self.Lr_Model),("Dtc",self.Dtc_Model),("Gnb",self.Gnb_Model)]
        self.Stacking_Model=StackingClassifier(estimators=self.stacking_estimator,final_estimator=self.Lr_Model)

        self.Lr_Grid=GridSearchCV(self.Lr_Model,self.Lr_Parameters)
        self.Gnb_Grid=GridSearchCV(self.Gnb_Model,self.Gnb_Parameters)

    def Create_Result_Table(self,List_Of_Result_Table_Columns):
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
        self.Result_Table=pd.DataFrame(columns=List_Of_Result_Table_Columns)

    def X_Y_df(self,Target):
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
        self.df_X=self.df.drop(Target,axis=1)
        self.df_Y=self.df[Target]

    def SKFold(self,n_splits,shuffle,random_state):
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
        self.KFold=StratifiedKFold(n_splits=n_splits,shuffle=shuffle,random_state=random_state)

    def algo_calculation(self,List_Model,List_Col):
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
            model.fit(self.train_X_os,self.train_Y_os)
            self.test_Pred=model.predict(self.test_X)
            Score=f1_score(self.test_Y,self.test_Pred)
            Score_List.append(Score)
        df_New_Row=pd.DataFrame([Score_List],columns=List_Col)
        self.Result_Table=self.Result_Table.append(df_New_Row,ignore_index=True)

    def Ave_Score(self,List_Column):
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
        for col in List_Column:
            Score=self.Result_Table[col].mean()
            Score_List.append(Score)
        self.Ave_Score_df=pd.DataFrame([Score_List],columns=List_Column)
        self.Result_Table=self.Result_Table.append(self.Ave_Score_df,ignore_index=True)
    
    def KFold_For_Loop(self,List_Column):
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
        for train_index, test_index in self.KFold.split(self.df_X,self.df_Y):
            self.train_X , self.test_X=self.df_X.iloc[train_index],self.df_X.iloc[test_index]
            self.train_Y , self.test_Y=self.df_Y.iloc[train_index],self.df_Y.iloc[test_index]
            self.ROS()
            self.algo_calculation([self.Lr_Grid,self.Bagging_Model,self.Boosting_Model,self.Stacking_Model],List_Column)

    def ROS(self):
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
        self.ros_model=RandomOverSampler(random_state=0)
        self.train_X_os , self.train_Y_os=self.ros_model.fit_resample(self.train_X,self.train_Y)