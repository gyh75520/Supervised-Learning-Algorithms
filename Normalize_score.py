# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:36:32 2017

@author: Vne
"""
import pandas as pd #数据分析
import numpy as np #科学计算

def Normalize_function(baseLine_score,bayesOptimal_score):
    f = lambda x: (x-baseLine_score)/(bayesOptimal_score-baseLine_score)
    return f
    
def getBayesOptimal_score(Summary_score):
    BayesOptimal_score = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score', 'RMSE','MXE'])
    a = Summary_score[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']].apply(lambda x: x.max())
    b = Summary_score[['RMSE','MXE']].apply(lambda x: x.min())
    a = a.to_frame().T
    b = b.to_frame().T
    BayesOptimal_score = pd.merge(a,b,left_index=True,right_index=True,how='outer')    
    return BayesOptimal_score
    
def getLowest_score(Summary_score):
    Lowest_score = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score', 'RMSE','MXE'])
    a = Summary_score[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']].apply(lambda x: x.min())
    b = Summary_score[['RMSE','MXE']].apply(lambda x: x.max())
    a = a.to_frame().T
    b = b.to_frame().T
    Lowest_score = pd.merge(a,b,left_index=True,right_index=True,how='outer')    
    return Lowest_score
    
def isBaseLine(p,BaseLine,Lowest):
    if p == 'RMSE' or p == 'MXE':
        return BaseLine>Lowest
    else:
        return BaseLine<Lowest
    
def Normalizing_score(Summary_score,BaseLine_score):
    Normalize_score = Summary_score.copy()
    BayesOptimal_score = getBayesOptimal_score(Summary_score)
    Lowest_score = getLowest_score(Summary_score)
    print (BayesOptimal_score)
    performances = np.array(BaseLine_score.columns)
    for p in performances:
        print (BaseLine_score[p][0],Lowest_score[p][0])
        if isBaseLine(p,BaseLine_score[p][0],Lowest_score[p][0]):
            f = Normalize_function(BaseLine_score[p][0],BayesOptimal_score[p][0])
        else:
            f = Normalize_function(Lowest_score[p][0],BayesOptimal_score[p][0])
        Normalize_score[p] = Normalize_score[p].apply(f)
    return Normalize_score

def AddBestToSummary(Summary_score,Classifier_name):
    if Classifier_name == 'SVM':
        platt_score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_platt_final.csv",index_col = 0)
        isotonic_score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_isotonic_final.csv",index_col = 0)
        Summary_score = Summary_score.append(pd.Series(platt_score.loc['best'],name=Classifier_name+'_platt'+'_best'))
        Summary_score = Summary_score.append(pd.Series(isotonic_score.loc['best'],name=Classifier_name+'_isotonic'+'_best'))
    else:
        score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_final.csv",index_col = 0)
        platt_score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_platt_final.csv",index_col = 0)
        isotonic_score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_isotonic_final.csv",index_col = 0)
        Summary_score = Summary_score.append(pd.Series(score.loc['best'],name= Classifier_name+'_best'))
        Summary_score = Summary_score.append(pd.Series(platt_score.loc['best'],name=Classifier_name+'_platt'+'_best'))
        Summary_score = Summary_score.append(pd.Series(isotonic_score.loc['best'],name=Classifier_name+'_isotonic'+'_best'))
    return Summary_score
    
def AddOPTToSummary(Summary_score,Classifier_name):
    if Classifier_name == 'SVM':
        platt_score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_platt_final.csv",index_col = 0)
        isotonic_score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_isotonic_final.csv",index_col = 0)
        Summary_score = Summary_score.append(pd.Series(platt_score.loc['OPT-SEL'],name=Classifier_name+'_platt'+'_OPT-SEL'))
        Summary_score = Summary_score.append(pd.Series(isotonic_score.loc['OPT-SEL'],name=Classifier_name+'_isotonic'+'_OPT-SEL'))
    else:    
        score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_final.csv",index_col = 0)
        platt_score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_platt_final.csv",index_col = 0)
        isotonic_score = pd.read_csv("E:\machine_learing\Result\\"+Classifier_name+"_isotonic_final.csv",index_col = 0)
        Summary_score = Summary_score.append(pd.Series(score.loc['OPT-SEL'],name= Classifier_name +'_OPT-SEL'))
        Summary_score = Summary_score.append(pd.Series(platt_score.loc['OPT-SEL'],name=Classifier_name+'_platt'+'_OPT-SEL'))
        Summary_score = Summary_score.append(pd.Series(isotonic_score.loc['OPT-SEL'],name=Classifier_name+'_isotonic'+'_OPT-SEL'))
    return Summary_score
    
def getBestSummary_score():
    #得分汇总
    Summary_score = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score', 'RMSE','MXE'])
    Summary_score = AddBestToSummary(Summary_score,'RandomForest')
    Summary_score = AddBestToSummary(Summary_score,'LogisticRegression')
    Summary_score = AddBestToSummary(Summary_score,'KNN')
    Summary_score = AddBestToSummary(Summary_score,'Naive_Bayes')
    Summary_score = AddBestToSummary(Summary_score,'NeuralNetwork')
    Summary_score = AddBestToSummary(Summary_score,'DecisionTree')
    Summary_score = AddBestToSummary(Summary_score,'SVM')
    Summary_score = AddBestToSummary(Summary_score,'Adaboost')
    Summary_score.to_csv('E:\machine_learing\Result\\Summary_best_score.csv')
    return Summary_score
    
def getOPTSummary_score():
    #得分汇总
    Summary_score = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score', 'RMSE','MXE'])
    Summary_score = AddOPTToSummary(Summary_score,'RandomForest')
    Summary_score = AddOPTToSummary(Summary_score,'LogisticRegression')
    Summary_score = AddOPTToSummary(Summary_score,'KNN')
    Summary_score = AddOPTToSummary(Summary_score,'Naive_Bayes')
    Summary_score = AddOPTToSummary(Summary_score,'NeuralNetwork')
    Summary_score = AddOPTToSummary(Summary_score,'DecisionTree')
    Summary_score = AddOPTToSummary(Summary_score,'SVM')
    Summary_score = AddOPTToSummary(Summary_score,'Adaboost')
    Summary_score.to_csv('E:\machine_learing\Result\\Summary_OPT_score.csv')
    return Summary_score

def getNormalizedScore():
    #BaseLine_score
    BaseLine_score = pd.read_csv("E:\machine_learing\Result\\BaseLine_score.csv",index_col = 0)
    BestSummary_score = getBestSummary_score()
    OPTSummary_score = getOPTSummary_score()
    Summary_score = pd.concat([BestSummary_score,OPTSummary_score])
    Best_length = len(BestSummary_score)
    Summary_length = len(Summary_score)
    Normalize_score = Normalizing_score(Summary_score,BaseLine_score)
    Normalize_score['Mean'] = Normalize_score.apply(lambda x: x.mean(),axis = 1)
    Normalize_best_score = Normalize_score.iloc[0:Best_length,:]
    Normalize_OPT_score = Normalize_score.iloc[Best_length:Summary_length,:]
    Normalize_best_score.to_csv('E:\machine_learing\Result\\Normalize_best_score.csv')
    Normalize_OPT_score.to_csv('E:\machine_learing\Result\\Normalize_OPT_score.csv')
    return Normalize_best_score,Normalize_OPT_score
    
def getNormalizedScoreforDsExp(Summary_score,BaseLine_score,number):
    Normalize_score = Normalizing_score(Summary_score,BaseLine_score)
    Normalize_score['Mean'] = Normalize_score.apply(lambda x: x.mean(),axis = 1)
    Normalize_score_arr = []
    length = int(len(Normalize_score)/number)
    n_range = range(0,number)
    for n in n_range:
        start = n*length
        end = start + length
        Normalize_best_score = Normalize_score.iloc[start:end,:]
        Normalize_score_arr.append(Normalize_best_score)
    return Normalize_score_arr
    
