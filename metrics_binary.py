# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:38:47 2017

@author: Vne
"""
from sklearn.metrics import accuracy_score, average_precision_score, f1_score,roc_auc_score,precision_score, recall_score,classification_report,precision_recall_curve,mean_squared_error,log_loss
import pandas as pd #数据分析
import numpy as np #科学计算
##求BEP
def getBEP(x1,y1,x2,y2):
    if x2==x1:
        return x1
    k=(y2-y1)/(x2-x1)
    x=(y2-k*x2)/(1-k)
    return x

##P-R曲线   
def getPRcurve(y, Yprobas_pred):
    precision, recall ,threshold= precision_recall_curve(y, Yprobas_pred,pos_label = 1)
    row_range = range(0,len(precision))
    global n_nbs
    for row in row_range:
        if precision[row]>=recall[row]:
            BEP_score = getBEP(recall[row-1],precision[row-1],recall[row],precision[row])
            print (BEP_score)
            #plt.plot(recall, precision,label='BEP_score='+str(BEP_score))
            return BEP_score

##求Average Per-class Accuracy         
def getAPaccuracy(y, yPred):
    length = len(yPred)
    l_range = range(0,length)
    class0_true_size = 0
    class0_size = 0       
    class1_true_size = 0
    class1_size = 0
    for l in l_range:
        if y[l]:
            class1_size+=1
            if yPred[l]:
                class1_true_size+=1
        else:
             class0_size+=1
             if yPred[l]==0:
                class0_true_size+=1
    print (class0_size,class0_true_size,class1_size,class1_true_size)
    APaccuracy =(class0_true_size/class0_size+class1_true_size/class1_size)/2
    return APaccuracy

def getScores(yPred,Yprobas_pred, y):
    accuracy = accuracy_score(y, yPred)
    average_precision = average_precision_score(y, Yprobas_pred)
    f1 = f1_score(y, yPred)
    roc_auc = roc_auc_score(y, Yprobas_pred)
    # convert from MSE to RMSE
    RMSE = np.sqrt(mean_squared_error(y, yPred))
    return (accuracy, average_precision, f1, roc_auc, RMSE)

def binary_scorer(estimator, x, y):
    predict_proba = estimator.predict_proba(x)
    Yprobas_pred = predict_proba[:,1]
    yPred = estimator.predict(x)
    accuracy, average_precision, f1, roc_auc, RMSE= getScores(yPred,Yprobas_pred,y)
    BEP_score = getPRcurve(y, Yprobas_pred)
    APaccuracy = getAPaccuracy(y, yPred)
    MXE = log_loss(y, Yprobas_pred)
    print ("mxe", MXE)
    print ('mean_score',(accuracy+average_precision+f1+roc_auc+BEP_score+RMSE+APaccuracy)/7)
    return (accuracy, average_precision, f1, roc_auc, RMSE,MXE,APaccuracy,BEP_score)

def getBaseLine_score(y_raw_train):
    score_arr = []
    length = len(y_raw_train)
    postive = 0
    for y_t in y_raw_train:
        if y_t:
            postive+=1
    y = y_raw_train
    postive_point= postive/float(length)
    if postive_point>=0.5:
        yPred = np.full(length, 1)
    if postive_point<0.5:
        yPred = np.full(length, 0)
    Yprobas_pred = np.full(length, postive_point)
    
    accuracy = accuracy_score(y, yPred)
    average_precision = average_precision_score(y, Yprobas_pred)
    f1 = f1_score(y, yPred)
    roc_auc = roc_auc_score(y, Yprobas_pred)
    # convert from MSE to RMSE
    RMSE = np.sqrt(mean_squared_error(y, yPred))
    BEP_score = getPRcurve(y, Yprobas_pred)
    APaccuracy = getAPaccuracy(y, yPred)
    MXE = log_loss(y, Yprobas_pred)
    
    score = [accuracy, average_precision, f1, roc_auc,APaccuracy,BEP_score, RMSE,MXE]
    score_arr.append(score)
    BaseLine_score = pd.DataFrame(np.array(score_arr),columns=['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score', 'RMSE','MXE'])
    BaseLine_score.to_csv('E:\machine_learing\Result\\BaseLine_score.csv')
    return BaseLine_score