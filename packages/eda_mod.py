import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def univariate_num(df_all_data,list_num_col,row,col):
    """
    Create histogram to visualise 1 continuous feature

    Parameters
    ----------
    df : dataframe
        dataframe to be inspected
    list : list
        list of continuous features
    row : int
        number of rows in figure
    col : int
        number of columns in figure
    
    Returns
    -------
    None
    """
    fig, ax=plt.subplots(row,col,figsize=(15,15))
    ax=ax.ravel()
    for i,feature in enumerate(list_num_col):
        ax[i].hist(df_all_data[feature],bins=50,edgecolor="white",density=True)
        ax[i].set_title(feature)
    for i,feature in enumerate(list_num_col):
        ax[i+4].hist(df_all_data[df_all_data["stroke"]==1][feature],bins=50,edgecolor="white",density=True)
        ax[i+4].set_title(feature)
    plt.show()

def univariate_cat(df_all_data,list_cat_col,row,col):
    """
    Create barplot to visualise frequency of all items in each categorical feature

    Parameters
    ----------
    df : dataframe
        dataframe to be inspected
    list : list
        list of categorical features
    row : int
        number of rows in figure
    col : int
        number of columns in figure
    
    Returns
    -------
    None
    """
    fig, ax=plt.subplots(row,col,figsize=(15,15))
    ax=ax.ravel()
    for i,feature in enumerate(list_cat_col):
        ax[i].bar(df_all_data[feature].value_counts().index,df_all_data[feature].value_counts().values)
        ax[i].set_title(feature)
        ax[i].set_xticks(range(len(df_all_data[feature].value_counts().index)))
        ax[i].set_xticklabels(df_all_data[feature].value_counts().index,rotation=45)
    for i,feature in enumerate(list_cat_col):
        ax[i+8].bar(df_all_data[df_all_data["stroke"]==1][feature].value_counts().index,df_all_data[df_all_data["stroke"]==1][feature].value_counts().values)
        ax[i+8].set_title(feature)
        ax[i+8].set_xticks(range(len(df_all_data[df_all_data["stroke"]==1][feature].value_counts().index)))
        ax[i+8].set_xticklabels(df_all_data[df_all_data["stroke"]==1][feature].value_counts().index,rotation=45)
    plt.show()

def bivariate_num(self,row,col):
    """
    Create boxplot to visualise distribution of feature wrt stroke

    Parameters
    ----------
    df : dataframe
        dataframe to be inspected
    list : list
        list of continuous features
    row : int
        number of rows in figure
    col : int
        number of columns in figure
    
    Returns
    -------
    None
    """
    fig, ax= plt.subplots(row,col,figsize=(15,15))
    ax=ax.ravel()
    for i,feature in enumerate(self.Num_Col):
        sns.boxplot(data=self.df,x="stroke",y=feature,ax=ax[i])
    for i,feature in enumerate(self.Num_Col):
        sns.boxplot(data=self.stroke_df,x="stroke",y=feature,ax=ax[i+4])
    plt.show()

def bivariate_cat(self,row,col):
    """
    Create countplot to visualise distribution of feature wrt stroke

    Parameters
    ----------
    df : dataframe
        dataframe to be inspected
    list : list
        list of continuous features
    row : int
        number of rows in figure
    col : int
        number of columns in figure
    
    Returns
    -------
    None
    """
    fig, ax= plt.subplots(row,col,figsize=(15,15))
    ax=ax.ravel()
    for i,feature in enumerate(self.Cat_Col):
        ax[i]=sns.countplot(data=self.df,x=feature,ax=ax[i])
    for i,feature in enumerate(self.Cat_Col):
        ax[i+8]=sns.countplot(data=self.stroke_df,x=feature,ax=ax[i+8])
    plt.show()