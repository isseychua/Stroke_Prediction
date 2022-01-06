import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def univariate_num(df,list,row,col):
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
    for i,feature in enumerate(list):
        ax[i].hist(df[feature],bins=50,edgecolor="white")
        ax[i].set_title(feature)
    plt.show()

def univariate_cat(df,list,row,col):
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
    for i,feature in enumerate(list):
        ax[i].bar(df[feature].value_counts().index,df[feature].value_counts().values)
        ax[i].set_title(feature)
        ax[i].set_xticks(range(len(df[feature].value_counts().index)))
        ax[i].set_xticklabels(df[feature].value_counts().index,rotation=45)
    plt.show()

def bivariate_num(df,list,row,col):
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
    for i,feature in enumerate(list):
        sns.boxplot(data=df,x="stroke",y=feature,ax=ax[i])
    plt.show()

def bivariate_cat(df,list,row,col):
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
    fig, ax= plt.subplots(2,4,figsize=(15,15))
    ax=ax.ravel()
    for i,feature in enumerate(list):
        ax[i]=sns.countplot(data=df,x=feature,hue="stroke",ax=ax[i])
    plt.show()