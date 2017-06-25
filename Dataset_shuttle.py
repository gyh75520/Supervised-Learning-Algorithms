# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:10:27 2017

@author: Vne
"""

import re
import pandas as pd #数据分析
import numpy as np #科学计算

from sklearn.model_selection import train_test_split

from basic_tool import scaler_attr
from basic_tool import describe_factor
from basic_tool import cvBestScore


data = pd.read_csv('E:\machine_learing\DateSet\shuttle.csv',index_col = 0)
data = data.dropna()
data = data[data.Shuttle != 3]
data = data[data.Shuttle != 2]
data['Shuttle'] = data['Shuttle'].replace(4,2)
data['Shuttle'] = data['Shuttle'].replace(5,3)
X = data.iloc[:,1:]
y = data['Shuttle']

##将样本划分为训练集和测试集
X_raw_train, X_test, y_raw_train, y_test = train_test_split(X, y, test_size=(len(data)-6500), random_state=1)


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

mean_cv_scores_df , mean_final_scores_df = SVM_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = SVM_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_isotonic_final.csv')
'''
mean_cv_scores_df , mean_final_scores_df = SVM_Classifier_Mul(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_final.csv')
'''
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
#setBaseLine_score
BaseLine_score = getMulBaseLine_score(y_raw_train,3)

from Normalize_score import getNormalizedScore
#getNormalizedScore
Normalize_best_score,Normalize_OPT_score = getNormalizedScore()
