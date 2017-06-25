# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:03:52 2017

@author: Vne
"""
import pandas as pd #数据分析
import numpy as np #科学计算
import sklearn.preprocessing as preprocessing

def describe_factor(x):
    ret = dict()
    for lvl in x.unique():
        if pd.isnull(lvl):
            ret["NaN"] = x.isnull().sum()
        else:
           ret[lvl] = np.sum(x==lvl)
    return ret
    
def scaler_attr(data,attr):
    scaler = preprocessing.StandardScaler()
    temp = np.array(data[attr]).reshape((len(data[attr]), 1))
    param = scaler.fit(temp)
    data[attr+'_scaled'] = scaler.fit_transform(temp, param)
    
def cvBestScore(mean_cv_scores_df,mean_final_scores_df):
    best = []
    performances = np.array(mean_cv_scores_df.columns)
    for p in performances:
        a = mean_cv_scores_df[p]['best']
        print (a)
        print (mean_final_scores_df[p][a])
        best.append(mean_final_scores_df[p][a])
    mean_final_scores_df.loc['best'] = best