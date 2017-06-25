# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 00:00:49 2017

@author: Vne
"""

import re
import pandas as pd #数据分析
import numpy as np #科学计算

from sklearn.model_selection import train_test_split

from basic_tool import scaler_attr
from basic_tool import describe_factor
from basic_tool import cvBestScore


def WinToNumber(win):
    if win == 'draw': 
        return 1
    if win == 'loss': 
        return 2
    if win == 'win': 
        return 3

        
data = pd.read_csv("E:\machine_learing\DateSet\connect-4.csv")
data.win = data.win.apply(WinToNumber)

b = pd.get_dummies(data['b'],prefix='b')
b1 = pd.get_dummies(data['b.1'],prefix='b.1')
b11 = pd.get_dummies(data['b.11'],prefix='b.11')
b21 = pd.get_dummies(data['b.21'],prefix='b.21')
b31 = pd.get_dummies(data['b.31'],prefix='b.31')
o = pd.get_dummies(data['o'],prefix='o')
o1 = pd.get_dummies(data['o.1'],prefix='o.1')
x = pd.get_dummies(data['x'],prefix='x')
x1 = pd.get_dummies(data['x.1'],prefix='x.1')

data = pd.merge(data,b,left_index=True,right_index=True,how='outer')
data = pd.merge(data,b1,left_index=True,right_index=True,how='outer')
data = pd.merge(data,b11,left_index=True,right_index=True,how='outer')
data = pd.merge(data,b21,left_index=True,right_index=True,how='outer')
data = pd.merge(data,b31,left_index=True,right_index=True,how='outer')
data = pd.merge(data,o,left_index=True,right_index=True,how='outer')
data = pd.merge(data,o1,left_index=True,right_index=True,how='outer')
data = pd.merge(data,x,left_index=True,right_index=True,how='outer')
data = pd.merge(data,x1,left_index=True,right_index=True,how='outer')

X = data.iloc[:,43:]
y = data['win']

##将样本划分为训练集和测试集
X_raw_train, X_test, y_raw_train, y_test = train_test_split(X, y, test_size=(len(data)-2000), random_state=1)


X_raw_train = X_raw_train.as_matrix()
y_raw_train = y_raw_train.as_matrix()
X_test = X_test.as_matrix()
y_test = y_test.as_matrix()

from mul_Classifier import RandomForest_Classifier_Mul
from mul_Classifier import KNN_Classifier_Mul
from mul_Classifier import LogisticRegression_Classifier_Mul
from mul_Classifier import DecisionTree_Classifier_Mul
from mul_Classifier import NeuralNetwork_Classifier_Mul
from mul_Classifier import Naive_Bayes_Classifier_Mul
from mul_Classifier import SVM_Classifier_Mul
from mul_Classifier import Adaboost_Classifier_Mul
'''
mean_cv_scores_df , mean_final_scores_df = RandomForest_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_final.csv')
mean_cv_scores_df , mean_final_scores_df = RandomForest_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = RandomForest_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = KNN_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\KNN_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\KNN_final.csv')
mean_cv_scores_df , mean_final_scores_df = KNN_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\KNN_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\KNN_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = KNN_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\KNN_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\KNN_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = LogisticRegression_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_final.csv')
mean_cv_scores_df , mean_final_scores_df = LogisticRegression_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = LogisticRegression_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = DecisionTree_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_final.csv')
mean_cv_scores_df , mean_final_scores_df = DecisionTree_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = DecisionTree_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = NeuralNetwork_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_final.csv')
mean_cv_scores_df , mean_final_scores_df = NeuralNetwork_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = NeuralNetwork_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = Naive_Bayes_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_final.csv')
mean_cv_scores_df , mean_final_scores_df = Naive_Bayes_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = Naive_Bayes_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_isotonic_final.csv')
'''
'''
mean_cv_scores_df , mean_final_scores_df = SVM_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_final.csv')
'''
mean_cv_scores_df , mean_final_scores_df = SVM_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = SVM_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = Adaboost_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_final.csv')
mean_cv_scores_df , mean_final_scores_df = Adaboost_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = Adaboost_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_isotonic_final.csv')


from metrics_mul import getMulBaseLine_score
#setBaseLine_score 需要 分类的数目 egg.mul_n = 5
BaseLine_score = getMulBaseLine_score(y_raw_train,3)

from Normalize_score import getNormalizedScore
#getNormalizedScore
Normalize_best_score,Normalize_OPT_score = getNormalizedScore()

