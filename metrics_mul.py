# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:35:33 2017

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
def getMulAPaccuracy(y, yPred,mul_n):
    length = len(yPred)
    l_range = range(0,length)
    true_size = np.full(mul_n, 0)
    size = np.full(mul_n, 0)       
    n_range = range(0,mul_n)
    for l in l_range:
        for n in n_range:
            if y[l] == n+1:
                size[n]+=1
                if yPred[l] == y[l]:
                    true_size[n]+=1
                break
                
    print (true_size,size)
    accuracy_sum = 0.0
    for n in n_range:
        accuracy_sum += float(true_size[n]/size[n])
    APaccuracy =  accuracy_sum/mul_n
    return APaccuracy

def mul_y(y,mul_n):
    res_arr = []
    size_y = len(y) 
    proportion_arr = []
    n_range = range(1,mul_n+1)
    for n in n_range :
        res = y.copy()
        r_range = range(0,len(res))
        number = 0;
        for r in r_range :
            if res[r] != n:
                res[r] = 0
            else:
                res[r] = 1
                number+=1
        res_arr.append(res)
        proportion = float(number/size_y)
        proportion_arr.append(proportion)
    return (res_arr,proportion_arr)
    
def mul_score(estimator, x, y):
    yPred = estimator.predict(x)
    predict_proba = estimator.predict_proba(x)
    
    mul_n = len(predict_proba.T)
    y_arr,proportion_arr = mul_y(y,mul_n)
    
    accuracy = accuracy_score(y, yPred)
    f1 = f1_score(y, yPred, average='weighted')
    # convert from MSE to RMSE
    RMSE = np.sqrt(mean_squared_error(y, yPred))
    APaccuracy = getMulAPaccuracy(y, yPred,mul_n)
    average_precision = 0
    BEP_score = 0
    MXE = 0
    roc_auc = 0
    
    m_range = range(0,mul_n)
    for m in m_range:
        Yprobas_pred = predict_proba[:,m]
        y = y_arr[m]
        average_precision = average_precision+proportion_arr[m]*average_precision_score(y, Yprobas_pred)
        BEP_score = BEP_score+proportion_arr[m]*getPRcurve(y, Yprobas_pred)
        MXE = MXE+proportion_arr[m]*log_loss(y, Yprobas_pred)
        roc_auc = roc_auc+proportion_arr[m]*roc_auc_score(y, Yprobas_pred)
        print ("B mxe"+str(m), average_precision,BEP_score,MXE,roc_auc)
    print ('mean',accuracy, average_precision, f1, roc_auc,APaccuracy,BEP_score, RMSE,MXE)
    return (accuracy, average_precision, f1, roc_auc, RMSE,MXE,APaccuracy,BEP_score)

#需要 分类的数目 egg.mul_n = 5
def getMulBaseLine_score(y_raw_train,mul_n):
    score_arr = []
    length = len(y_raw_train)
    
    accuracy = 0
    f1 = 0
    RMSE = 0
    APaccuracy = 0
    average_precision = 0
    BEP_score = 0
    MXE = 0
    roc_auc = 0
    
    y_arr,proportion_arr = mul_y(y_raw_train,mul_n)
    y = y_raw_train
    m_range = range(0,mul_n)
    for m in m_range:
         yPred = np.full(length, m+1)
         Yprobas_pred = np.full(length, proportion_arr[m])
         accuracy = accuracy+proportion_arr[m]*accuracy_score(y, yPred)
         RMSE = RMSE + proportion_arr[m]*np.sqrt(mean_squared_error(y, yPred))
         f1 = f1 + proportion_arr[m]*f1_score(y, yPred ,average='weighted')
         print ('f1',f1_score(y, yPred ,average='weighted'))
         APaccuracy = APaccuracy+proportion_arr[m]*getMulAPaccuracy(y, yPred,mul_n)
         average_precision = average_precision+proportion_arr[m]*average_precision_score(y_arr[m], Yprobas_pred)
         BEP_score = BEP_score+proportion_arr[m]*getPRcurve(y_arr[m], Yprobas_pred)
         MXE = MXE+proportion_arr[m]*log_loss(y_arr[m], Yprobas_pred)
         roc_auc = roc_auc+proportion_arr[m]*roc_auc_score(y_arr[m], Yprobas_pred)
    
    score = [accuracy, average_precision, f1, roc_auc,APaccuracy,BEP_score, RMSE,MXE]
    score_arr.append(score)
    BaseLine_score = pd.DataFrame(np.array(score_arr),columns=['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score', 'RMSE','MXE'])
    BaseLine_score.to_csv('E:\machine_learing\Result\\BaseLine_score.csv')
    return BaseLine_score
    