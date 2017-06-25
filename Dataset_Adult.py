# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import pandas as pd #数据分析
import numpy as np #科学计算

from sklearn.model_selection import train_test_split

from basic_tool import scaler_attr
from basic_tool import describe_factor
from basic_tool import cvBestScore

data = pd.read_csv("E:\machine_learing\DateSet\Adult.csv")
data = data[data.occupation != '?']
raw_data = data[data.occupation != '?']

data['workclass_num'] = data.workclass.map({'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
data['over50K'] = np.where(data.income == '<=50K', 0, 1)
data['marital_num'] = data['marital.status'].map({'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3, 'Married-civ-spouse':4, 'Married-AF-spouse':4, 'Married-spouse-absent':5})
data['race_num'] = data.race.map({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
data['sex_num'] = np.where(data.sex == 'Female', 0, 1)
data['rel_num'] = data.relationship.map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})

scaler_attr(data,'capital.gain')
scaler_attr(data,'capital.loss')

X = data[['workclass_num', 'education.num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital.gain_scaled', 'capital.loss_scaled']]
y = data.over50K

##将样本划分为训练集和测试集
X_raw_train, X_test, y_raw_train, y_test = train_test_split(X, y, test_size=(len(data)-4000), random_state=1)


X_raw_train = X_raw_train.as_matrix()
y_raw_train = y_raw_train.as_matrix()
X_test = X_test.as_matrix()
y_test = y_test.as_matrix()

from binary_Classifier import RandomForest_Classifier
from binary_Classifier import KNN_Classifier
from binary_Classifier import LogisticRegression_Classifier
from binary_Classifier import DecisionTree_Classifier
from binary_Classifier import NeuralNetwork_Classifier
from binary_Classifier import Naive_Bayes_Classifier
from binary_Classifier import SVM_Classifier
from binary_Classifier import Adaboost_Classifier


mean_cv_scores_df , mean_final_scores_df = RandomForest_Classifier(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_final.csv')
mean_cv_scores_df , mean_final_scores_df = RandomForest_Classifier(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = RandomForest_Classifier(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\RandomForest_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = KNN_Classifier(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\KNN_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\KNN_final.csv')
mean_cv_scores_df , mean_final_scores_df = KNN_Classifier(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\KNN_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\KNN_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = KNN_Classifier(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\KNN_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\KNN_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = LogisticRegression_Classifier(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_final.csv')
mean_cv_scores_df , mean_final_scores_df = LogisticRegression_Classifier(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = LogisticRegression_Classifier(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\LogisticRegression_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = DecisionTree_Classifier(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_final.csv')
mean_cv_scores_df , mean_final_scores_df = DecisionTree_Classifier(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = DecisionTree_Classifier(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\DecisionTree_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = NeuralNetwork_Classifier(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_final.csv')
mean_cv_scores_df , mean_final_scores_df = NeuralNetwork_Classifier(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = NeuralNetwork_Classifier(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\NeuralNetwork_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = Naive_Bayes_Classifier(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_final.csv')
mean_cv_scores_df , mean_final_scores_df = Naive_Bayes_Classifier(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = Naive_Bayes_Classifier(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Naive_Bayes_isotonic_final.csv')
'''
mean_cv_scores_df , mean_final_scores_df = SVM_Classifier(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_final.csv')
'''
mean_cv_scores_df , mean_final_scores_df = SVM_Classifier(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = SVM_Classifier(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\SVM_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\SVM_isotonic_final.csv')

mean_cv_scores_df , mean_final_scores_df = Adaboost_Classifier(X_raw_train,y_raw_train,X_test, y_test,'')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_final.csv')
mean_cv_scores_df , mean_final_scores_df = Adaboost_Classifier(X_raw_train,y_raw_train,X_test, y_test,'platt')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_platt_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_platt_final.csv')
mean_cv_scores_df , mean_final_scores_df = Adaboost_Classifier(X_raw_train,y_raw_train,X_test, y_test,'isotonic')
cvBestScore(mean_cv_scores_df,mean_final_scores_df)
mean_cv_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_isotonic_cv.csv')
mean_final_scores_df.to_csv('E:\machine_learing\Result\\Adaboost_isotonic_final.csv')

from metrics_binary import getBaseLine_score
#setBaseLine_score
BaseLine_score = getBaseLine_score(y_raw_train)

from Normalize_score import getNormalizedScore
#getNormalizedScore
Normalize_best_score,Normalize_OPT_score = getNormalizedScore()
'''
#plt.plot(precision, recall)
plt.plot([0, 1], [0, 1],color="red",label='y=x')
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
Final_socre = Final_Experiment()
Final_socre.to_csv('E:\machine_learing\Result\\Final_socre.csv')

'''
