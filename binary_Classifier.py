# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:59:54 2017

@author: Vne
"""
import pandas as pd #数据分析
import numpy as np #科学计算

from sklearn.model_selection import KFold

from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from metrics_binary import binary_scorer

def KFold_Experiment(estimator,kf,X_raw_train,y_raw_train,X_test, y_test ,Cali_method):
    cv_score_arr = []
    final_score_arr = []
    for train_index, cv_index in kf.split(X_raw_train):
        X_train, X_cv = X_raw_train[train_index], X_raw_train[cv_index]
        y_train, y_cv = y_raw_train[train_index], y_raw_train[cv_index]
        train_len = len(X_train)
        cal_train_len = int(train_len*3/4)
        print ('len',cal_train_len)
        if Cali_method=='isotonic':
            print (Cali_method)
            #3k train and 1k cal
            estimator.fit(X_train[0:cal_train_len],y_train[0:cal_train_len])
            estimator_isotonic = CalibratedClassifierCV(estimator, cv='prefit', method='isotonic')
            estimator_isotonic.fit(X_train[cal_train_len:train_len],y_train[cal_train_len:train_len])
            cv_score = binary_scorer(estimator, X_cv, y_cv)
            final_score = binary_scorer(estimator, X_test, y_test)
        elif Cali_method=='platt':
            print (Cali_method)
            #3k train and 1k cal
            estimator.fit(X_train[0:cal_train_len],y_train[0:cal_train_len])
            estimator_platt = CalibratedClassifierCV(estimator, cv='prefit', method='sigmoid')
            estimator_platt.fit(X_train[cal_train_len:train_len],y_train[cal_train_len:train_len])
            cv_score = binary_scorer(estimator, X_cv, y_cv)
            final_score = binary_scorer(estimator, X_test, y_test)
        else :
            estimator.fit(X_train,y_train)
            cv_score = binary_scorer(estimator, X_cv, y_cv)
            final_score = binary_scorer(estimator, X_test, y_test)
        cv_score_arr.append(cv_score)
        final_score_arr.append(final_score)
        print("TRAIN:", train_index, "CV:", cv_index,"score",cv_score)
    return (np.array(cv_score_arr),np.array(final_score_arr))
    
def KNN_Classifier(X_raw_train,y_raw_train,X_test, y_test,Cali_method):
    k_range = range(1,30)       
    mean_cv_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    mean_final_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    weight=['uniform','distance']
    metrics=['euclidean','minkowski']
    index_name = []
    ##采用5次五折交叉验证法
    kf = KFold(n_splits=5)  
    for k in k_range:
        for w in weight:
            for m in metrics:
                knn = KNeighborsClassifier()
                knn.set_params(n_neighbors=k,weights=w,metric=m)
                name = 'n_neighbors='+str(k)+' weights='+w+' metric='+m
                index_name.append(name)
                cv_score,final_score = KFold_Experiment(knn,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
                #cv
                cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
                cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
                mean_cv_scores = cv_scores.loc['mean']
                mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
                #final
                final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
                final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
                mean_final_scores = final_scores.loc['mean']
                mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
                
    #cv    
    mean_cv_scores_df.index=index_name
    mean_cv_scores_dfmax = mean_cv_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_cv_scores_dfmax.loc['best'] = mean_cv_scores_dfmax.apply(lambda x: x.argmax()) 
    mean_cv_scores_dfmin = mean_cv_scores_df[['RMSE','MXE']]
    mean_cv_scores_dfmin.loc['best'] = mean_cv_scores_dfmin.apply(lambda x: x.argmin())
    mean_cv_scores_df = pd.merge(mean_cv_scores_dfmax,mean_cv_scores_dfmin,left_index=True,right_index=True,how='outer') 
    #final
    mean_final_scores_df.index=index_name
    mean_final_scores_dfmax = mean_final_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_final_scores_dfmax.loc['OPT-SEL'] = mean_final_scores_dfmax.apply(lambda x: x.max()) 
    mean_final_scores_dfmin = mean_final_scores_df[['RMSE','MXE']]
    mean_final_scores_dfmin.loc['OPT-SEL'] = mean_final_scores_dfmin.apply(lambda x: x.min())
    mean_final_scores_df = pd.merge(mean_final_scores_dfmax,mean_final_scores_dfmin,left_index=True,right_index=True,how='outer')
    return mean_cv_scores_df,mean_final_scores_df
    
def LogisticRegression_Classifier(X_raw_train,y_raw_train,X_test, y_test,Cali_method):
    mean_cv_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    mean_final_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    #c_range = range(10^(-8) 到 10^4)
    c_range = [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
    #‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty.
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    index_name = []
    ##采用5次五折交叉验证法
    kf = KFold(n_splits=5)  
    
    for c in c_range:
        for sol in solvers:
            lr = LogisticRegression()
            lr.set_params(C=c, penalty='l2',solver=sol)
            name = 'C='+str(c)+' penalty=l2'+' solver='+sol
            index_name.append(name)
            cv_score,final_score = KFold_Experiment(lr,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
            #cv
            cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
            cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
            mean_cv_scores = cv_scores.loc['mean']
            mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
            #final
            final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
            final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
            mean_final_scores = final_scores.loc['mean']
            mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
            
    for c in c_range:
        lr = LogisticRegression()
        lr.set_params(C=c, penalty='l1',solver='liblinear')
        name = 'C='+str(c)+' penalty=l1'+' solver=liblinear'
        index_name.append(name)
        cv_score,final_score = KFold_Experiment(lr,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
        #cv
        cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
        cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
        mean_cv_scores = cv_scores.loc['mean']
        mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
        #final
        final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
        final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
        mean_final_scores = final_scores.loc['mean']
        mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
        
    #cv    
    mean_cv_scores_df.index=index_name
    mean_cv_scores_dfmax = mean_cv_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_cv_scores_dfmax.loc['best'] = mean_cv_scores_dfmax.apply(lambda x: x.argmax()) 
    mean_cv_scores_dfmin = mean_cv_scores_df[['RMSE','MXE']]
    mean_cv_scores_dfmin.loc['best'] = mean_cv_scores_dfmin.apply(lambda x: x.argmin())
    mean_cv_scores_df = pd.merge(mean_cv_scores_dfmax,mean_cv_scores_dfmin,left_index=True,right_index=True,how='outer') 
    #final
    mean_final_scores_df.index=index_name
    mean_final_scores_dfmax = mean_final_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_final_scores_dfmax.loc['OPT-SEL'] = mean_final_scores_dfmax.apply(lambda x: x.max()) 
    mean_final_scores_dfmin = mean_final_scores_df[['RMSE','MXE']]
    mean_final_scores_dfmin.loc['OPT-SEL'] = mean_final_scores_dfmin.apply(lambda x: x.min())
    mean_final_scores_df = pd.merge(mean_final_scores_dfmax,mean_final_scores_dfmin,left_index=True,right_index=True,how='outer')
    return mean_cv_scores_df,mean_final_scores_df

def Naive_Bayes_Classifier(X_raw_train,y_raw_train,X_test, y_test,Cali_method):
    mean_cv_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    mean_final_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    index_name = []
    ##采用5次五折交叉验证法
    kf = KFold(n_splits=5)  
    #GaussianNB
    gnb = GaussianNB()
    name = 'GaussianNB'
    index_name.append(name)
    cv_score,final_score = KFold_Experiment(gnb,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
    #cv
    cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
    mean_cv_scores = cv_scores.loc['mean']
    mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
    #final
    final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
    mean_final_scores = final_scores.loc['mean']
    mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
    
    
    #平滑参数
    alpha_range = range(0,2)
    '''
    for al in alpha_range:
        #MultinomialNB
        mnb = MultinomialNB()
        mnb.set_params(alpha=al)
        name = 'MultinomialNB'+' alpha='+str(al)
        index_name.append(name)
        cv_score,final_score = KFold_Experiment(gnb,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
        #cv
        cv_scores = DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
        cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
        mean_scores = cv_scores.loc['mean']
        mean_cv_scores_df = mean_cv_scores_df.append(mean_scores,ignore_index=True)
        #final
        final_scores = DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
        cv_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
        final_scores = cv_scores.loc['mean']
        mean_final_scores_df = mean_final_scores_df.append(mean_scores,ignore_index=True)
    
    for al in alpha_range:
        #BernoulliNB
        bnb = BernoulliNB()
        bnb.set_params(alpha=al)
        name = 'BernoulliNB'+' alpha='+str(al)
        index_name.append(name)
        cv_score,final_score = KFold_Experiment(gnb,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
        #cv
        cv_scores = DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
        cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
        mean_scores = cv_scores.loc['mean']
        mean_cv_scores_df = mean_cv_scores_df.append(mean_scores,ignore_index=True)
        #final
        final_scores = DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
        cv_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
        final_scores = cv_scores.loc['mean']
        mean_final_scores_df = mean_final_scores_df.append(mean_scores,ignore_index=True)
    '''
    #cv    
    mean_cv_scores_df.index=index_name
    mean_cv_scores_dfmax = mean_cv_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_cv_scores_dfmax.loc['best'] = mean_cv_scores_dfmax.apply(lambda x: x.argmax()) 
    mean_cv_scores_dfmin = mean_cv_scores_df[['RMSE','MXE']]
    mean_cv_scores_dfmin.loc['best'] = mean_cv_scores_dfmin.apply(lambda x: x.argmin())
    mean_cv_scores_df = pd.merge(mean_cv_scores_dfmax,mean_cv_scores_dfmin,left_index=True,right_index=True,how='outer') 
    #final
    mean_final_scores_df.index=index_name
    mean_final_scores_dfmax = mean_final_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_final_scores_dfmax.loc['OPT-SEL'] = mean_final_scores_dfmax.apply(lambda x: x.max()) 
    mean_final_scores_dfmin = mean_final_scores_df[['RMSE','MXE']]
    mean_final_scores_dfmin.loc['OPT-SEL'] = mean_final_scores_dfmin.apply(lambda x: x.min())
    mean_final_scores_df = pd.merge(mean_final_scores_dfmax,mean_final_scores_dfmin,left_index=True,right_index=True,how='outer')
    return mean_cv_scores_df,mean_final_scores_df
    
def DecisionTree_Classifier(X_raw_train,y_raw_train,X_test, y_test,Cali_method):
    mean_cv_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    mean_final_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    index_name = []
    ##采用5次五折交叉验证法
    kf = KFold(n_splits=5)  
    criterions=['gini','entropy'] 
    for ct in criterions:
        dt = DecisionTreeClassifier()
        dt.set_params(criterion=ct)
        name = 'criterion='+ct
        index_name.append(name)
        cv_score,final_score = KFold_Experiment(dt,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
        #cv
        cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
        cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
        mean_cv_scores = cv_scores.loc['mean']
        mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
        #final
        final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
        final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
        mean_final_scores = final_scores.loc['mean']
        mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
    #cv    
    mean_cv_scores_df.index=index_name
    mean_cv_scores_dfmax = mean_cv_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_cv_scores_dfmax.loc['best'] = mean_cv_scores_dfmax.apply(lambda x: x.argmax()) 
    mean_cv_scores_dfmin = mean_cv_scores_df[['RMSE','MXE']]
    mean_cv_scores_dfmin.loc['best'] = mean_cv_scores_dfmin.apply(lambda x: x.argmin())
    mean_cv_scores_df = pd.merge(mean_cv_scores_dfmax,mean_cv_scores_dfmin,left_index=True,right_index=True,how='outer') 
    #final
    mean_final_scores_df.index=index_name
    mean_final_scores_dfmax = mean_final_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_final_scores_dfmax.loc['OPT-SEL'] = mean_final_scores_dfmax.apply(lambda x: x.max()) 
    mean_final_scores_dfmin = mean_final_scores_df[['RMSE','MXE']]
    mean_final_scores_dfmin.loc['OPT-SEL'] = mean_final_scores_dfmin.apply(lambda x: x.min())
    mean_final_scores_df = pd.merge(mean_final_scores_dfmax,mean_final_scores_dfmin,left_index=True,right_index=True,how='outer')
    return mean_cv_scores_df,mean_final_scores_df

def RandomForest_Classifier(X_raw_train,y_raw_train,X_test, y_test,Cali_method):
    mean_cv_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    mean_final_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    index_name = []
    ##采用5次五折交叉验证法
    kf = KFold(n_splits=5)
    n = 1024
    criterions=['gini','entropy']
    # max_features must be in (0, n_features] [1,2,4,6,8,12,16,20]
    max_feature = [1,2,4]
    for ct in criterions:
        for feature in max_feature:
            rf = RandomForestClassifier()
            rf.set_params(n_estimators=n,criterion=ct,max_features = feature)
            name = 'n_estimators='+str(n)+' criterion='+ct+' max_features'+str(feature)
            index_name.append(name)
            cv_score,final_score = KFold_Experiment(rf,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
            #cv
            cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
            cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
            mean_cv_scores = cv_scores.loc['mean']
            mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
            #final
            final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
            final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
            mean_final_scores = final_scores.loc['mean']
            mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
            
    #cv    
    mean_cv_scores_df.index=index_name
    mean_cv_scores_dfmax = mean_cv_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_cv_scores_dfmax.loc['best'] = mean_cv_scores_dfmax.apply(lambda x: x.argmax()) 
    mean_cv_scores_dfmin = mean_cv_scores_df[['RMSE','MXE']]
    mean_cv_scores_dfmin.loc['best'] = mean_cv_scores_dfmin.apply(lambda x: x.argmin())
    mean_cv_scores_df = pd.merge(mean_cv_scores_dfmax,mean_cv_scores_dfmin,left_index=True,right_index=True,how='outer') 
    #final
    mean_final_scores_df.index=index_name
    mean_final_scores_dfmax = mean_final_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_final_scores_dfmax.loc['OPT-SEL'] = mean_final_scores_dfmax.apply(lambda x: x.max()) 
    mean_final_scores_dfmin = mean_final_scores_df[['RMSE','MXE']]
    mean_final_scores_dfmin.loc['OPT-SEL'] = mean_final_scores_dfmin.apply(lambda x: x.min())
    mean_final_scores_df = pd.merge(mean_final_scores_dfmax,mean_final_scores_dfmin,left_index=True,right_index=True,how='outer')
    return mean_cv_scores_df,mean_final_scores_df
    
def NeuralNetwork_Classifier(X_raw_train,y_raw_train,X_test, y_test,Cali_method):
    mean_cv_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    mean_final_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    index_name = []
    hidden_layer = [1,2,4,8,32,64,128,256,512]
    activations = ['identity', 'logistic', 'tanh', 'relu']
    solvers = ['lbfgs','sgd', 'adam']
    ##采用5次五折交叉验证法
    kf = KFold(n_splits=5)
    for h in hidden_layer:
        for a in activations:
            for s in solvers:
                mlp = MLPClassifier()
                mlp.set_params(hidden_layer_sizes=h,activation=a,solver=s)
                name = 'hidden_layer_sizes='+str(h)+' activation='+a+' solvers='+s
                index_name.append(name)
                cv_score,final_score = KFold_Experiment(mlp,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
                #cv
                cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
                cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
                mean_cv_scores = cv_scores.loc['mean']
                mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
                #final
                final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
                final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
                mean_final_scores = final_scores.loc['mean']
                mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
                
    #cv    
    mean_cv_scores_df.index=index_name
    mean_cv_scores_dfmax = mean_cv_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_cv_scores_dfmax.loc['best'] = mean_cv_scores_dfmax.apply(lambda x: x.argmax()) 
    mean_cv_scores_dfmin = mean_cv_scores_df[['RMSE','MXE']]
    mean_cv_scores_dfmin.loc['best'] = mean_cv_scores_dfmin.apply(lambda x: x.argmin())
    mean_cv_scores_df = pd.merge(mean_cv_scores_dfmax,mean_cv_scores_dfmin,left_index=True,right_index=True,how='outer') 
    #final
    mean_final_scores_df.index=index_name
    mean_final_scores_dfmax = mean_final_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_final_scores_dfmax.loc['OPT-SEL'] = mean_final_scores_dfmax.apply(lambda x: x.max()) 
    mean_final_scores_dfmin = mean_final_scores_df[['RMSE','MXE']]
    mean_final_scores_dfmin.loc['OPT-SEL'] = mean_final_scores_dfmin.apply(lambda x: x.min())
    mean_final_scores_df = pd.merge(mean_final_scores_dfmax,mean_final_scores_dfmin,left_index=True,right_index=True,how='outer')
    return mean_cv_scores_df,mean_final_scores_df

def SVM_Classifier(X_raw_train,y_raw_train,X_test, y_test,Cali_method):
    mean_cv_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    mean_final_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    index_name = []
    ##采用5次五折交叉验证法
    kf = KFold(n_splits=5)
    #c_range = range(10^(-8) 到 10^4)
    c_range = [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
    #'linear','poly'暂时用不了
    kernels = ['rbf', 'sigmoid']
    for c in c_range:
        for k in kernels:
            svc = SVC( probability = True)
            svc.set_params(C=c,kernel=k)
            name = 'C='+str(c)+' kernel='+k
            index_name.append(name)
            cv_score,final_score = KFold_Experiment(svc,kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
            #cv
            cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
            cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
            mean_cv_scores = cv_scores.loc['mean']
            mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
            #final
            final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
            final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
            mean_final_scores = final_scores.loc['mean']
            mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
            
    #cv    
    mean_cv_scores_df.index=index_name
    mean_cv_scores_dfmax = mean_cv_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_cv_scores_dfmax.loc['best'] = mean_cv_scores_dfmax.apply(lambda x: x.argmax()) 
    mean_cv_scores_dfmin = mean_cv_scores_df[['RMSE','MXE']]
    mean_cv_scores_dfmin.loc['best'] = mean_cv_scores_dfmin.apply(lambda x: x.argmin())
    mean_cv_scores_df = pd.merge(mean_cv_scores_dfmax,mean_cv_scores_dfmin,left_index=True,right_index=True,how='outer') 
    #final
    mean_final_scores_df.index=index_name
    mean_final_scores_dfmax = mean_final_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_final_scores_dfmax.loc['OPT-SEL'] = mean_final_scores_dfmax.apply(lambda x: x.max()) 
    mean_final_scores_dfmin = mean_final_scores_df[['RMSE','MXE']]
    mean_final_scores_dfmin.loc['OPT-SEL'] = mean_final_scores_dfmin.apply(lambda x: x.min())
    mean_final_scores_df = pd.merge(mean_final_scores_dfmax,mean_final_scores_dfmin,left_index=True,right_index=True,how='outer')
    return mean_cv_scores_df,mean_final_scores_df
    
def Adaboost_Classifier(X_raw_train,y_raw_train,X_test, y_test,Cali_method):
    mean_cv_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    mean_final_scores_df = pd.DataFrame(columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
    index_name = []
    ##采用5次五折交叉验证法
    kf = KFold(n_splits=5)
    
    dt = DecisionTreeClassifier()
    gnb = GaussianNB()
    base_estimators = [{'name':'Naive_Bayes','estimator':gnb},{'name':'DecisionTree','estimator':dt}]
    n_estimator = [20,30,40,50,60,70,80,90,100]
    learning_rates = [0.2,0.4,0.6,0.8,1]
    for b in base_estimators:
        for n in n_estimator:
            for r in learning_rates:
                ada = AdaBoostClassifier(algorithm='SAMME')
                ada.set_params(base_estimator=b['estimator'],n_estimators=n,learning_rate=r)
                name = 'base_estimator='+b['name']+' n_estimators='+str(n)+' learning_rates='+str(r)
                index_name.append(name)
                cv_score,final_score = KFold_Experiment(b['estimator'],kf, X_raw_train, y_raw_train,X_test, y_test,Cali_method)
                #cv
                cv_scores = pd.DataFrame(cv_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
                cv_scores.loc['mean'] = cv_scores.apply(lambda x: x.mean())
                mean_cv_scores = cv_scores.loc['mean']
                mean_cv_scores_df = mean_cv_scores_df.append(mean_cv_scores,ignore_index=True)
                #final
                final_scores = pd.DataFrame(final_score,columns=['accuracy', 'average_precision', 'f1', 'roc_auc', 'RMSE','MXE','APaccuracy','BEP_score'])
                final_scores.loc['mean'] = final_scores.apply(lambda x: x.mean())
                mean_final_scores = final_scores.loc['mean']
                mean_final_scores_df = mean_final_scores_df.append(mean_final_scores,ignore_index=True)
                
    #cv    
    mean_cv_scores_df.index=index_name
    mean_cv_scores_dfmax = mean_cv_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_cv_scores_dfmax.loc['best'] = mean_cv_scores_dfmax.apply(lambda x: x.argmax()) 
    mean_cv_scores_dfmin = mean_cv_scores_df[['RMSE','MXE']]
    mean_cv_scores_dfmin.loc['best'] = mean_cv_scores_dfmin.apply(lambda x: x.argmin())
    mean_cv_scores_df = pd.merge(mean_cv_scores_dfmax,mean_cv_scores_dfmin,left_index=True,right_index=True,how='outer') 
    #final
    mean_final_scores_df.index=index_name
    mean_final_scores_dfmax = mean_final_scores_df[['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score']]
    mean_final_scores_dfmax.loc['OPT-SEL'] = mean_final_scores_dfmax.apply(lambda x: x.max()) 
    mean_final_scores_dfmin = mean_final_scores_df[['RMSE','MXE']]
    mean_final_scores_dfmin.loc['OPT-SEL'] = mean_final_scores_dfmin.apply(lambda x: x.min())
    mean_final_scores_df = pd.merge(mean_final_scores_dfmax,mean_final_scores_dfmin,left_index=True,right_index=True,how='outer')
    return mean_cv_scores_df,mean_final_scores_df

    