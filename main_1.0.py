#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold ,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier , AdaBoostClassifier , StackingClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE , RandomOverSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

from packages.preprocessing_mod import inspect_dataframe, fill_nan,check_outliers
from packages.eda_mod import univariate_num , univariate_cat , bivariate_cat , bivariate_num
from packages.modelling_mod import algo_calculation, Ave_Score

#Read csv file into dataframe
pd.set_option("display.max_columns",None)
df=pd.read_csv("Source/healthcare-dataset-stroke-data.csv")

#Inspect dataframe
inspect_dataframe(df)
print(df.stroke.value_counts(normalize=True))

#Split into categorical and numerical columns
Col=df.columns.tolist()
Num_Col=["id","age","avg_glucose_level","bmi"]
Cat_Col=["hypertension","heart_disease","gender","ever_married","work_type","Residence_type","smoking_status","stroke"]

#Check for nan and duplicated values
print(df.isnull().sum())
print(df.duplicated().sum())

#Fill nan values
df=fill_nan(df,["bmi"])

#Remove "id" from Num_Col
Num_Col_less_id=Num_Col
Num_Col_less_id.remove("id")

#Check and drop outliers
outlier_index_unique=check_outliers(df,Num_Col_less_id)
df=df.drop(outlier_index_unique,axis=0).reset_index(drop=True)
print(df.shape)

#Uni Variate analysis
univariate_num(df,Num_Col,2,2)
univariate_cat(df,Cat_Col,2,4)

#Bi Variate analysis
bivariate_num(df,Num_Col,2,2)
bivariate_cat(df,Cat_Col,2,4)

#Create get_dummies
sns.heatmap(df.corr())
plt.show()

#Feature Engineering
print(df.head())
encoder=preprocessing.LabelEncoder()
for col in ["gender","ever_married","work_type","Residence_type","smoking_status"]:
    df[col]=encoder.fit_transform(df[col])
print(df.head())
#Modelling
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

Results_Table=pd.DataFrame(columns=["Lr","Bagging","Boosting","Stacking"])

#Create train and test set
df_X=df.drop("stroke",axis=1)
df_Y=df["stroke"]

#Train_Test_Split followed by SMOTE oversampling
KFold=StratifiedKFold(n_splits=5,shuffle=True,random_state=1)
for train_index, test_index in KFold.split(df_X,df_Y):
    train_X , test_X=df_X.iloc[train_index],df_X.iloc[test_index]
    train_Y , test_Y=df_Y.iloc[train_index],df_Y.iloc[test_index]

    Ros=RandomOverSampler(random_state=0)
    train_X_os , train_Y_os=Ros.fit_resample(train_X,train_Y)
    
    New_Row=algo_calculation([Lr_Grid,Bagging_Model,Boosting_Model,Stacking_Model],train_X_os,train_Y_os,test_X,test_Y)
    df_New_Row=pd.DataFrame([New_Row],columns=["Lr","Bagging","Boosting","Stacking"])
    Results_Table=Results_Table.append(df_New_Row,ignore_index=True)

#Calculate the average f1 score for each method
Ave_Row=Ave_Score(Results_Table,["Lr","Bagging","Boosting","Stacking"])
df_Ave_Row=pd.DataFrame([Ave_Row],columns=["Lr","Bagging","Boosting","Stacking"])
Results_Table=Results_Table.append(df_Ave_Row,ignore_index=True)
print(Results_Table)

#Export table of results in csv
Results_Table.to_csv("results_ros.txt",index=False)

