# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:51:39 2017

@author: Vne
"""

import pandas as pd #数据分析
import numpy as np #科学计算

#pd.set_option('precision',20)

def getPerformance_score(Normalize_score_arr):
    length = len(Normalize_score_arr)
    accuracy = 0
    average_precision = 0
    f1 = 0
    roc_auc = 0
    APaccuracy = 0
    BEP_score = 0
    RMSE = 0
    MXE = 0
    for Normalize_score in Normalize_score_arr:
        accuracy+=Normalize_score['accuracy']
        average_precision+=Normalize_score['average_precision']
        f1+=Normalize_score['f1']
        roc_auc+=Normalize_score['roc_auc']
        APaccuracy+=Normalize_score['APaccuracy']
        BEP_score+=Normalize_score['BEP_score']
        RMSE+=Normalize_score['RMSE']
        MXE+=Normalize_score['MXE']
        
    accuracy = accuracy/length
    average_precision = average_precision/length
    f1 = f1/length
    roc_auc = roc_auc/length
    APaccuracy = APaccuracy/length
    BEP_score = BEP_score/length
    RMSE = RMSE/length
    MXE = MXE/length
    
    Performance_score = pd.concat([accuracy,average_precision,f1,roc_auc,APaccuracy,BEP_score,RMSE,MXE],axis = 1 )
    Performance_score['Mean'] = Performance_score.apply(lambda x: x.mean(),axis = 1)
    return Performance_score
    
def rankTable(algorithm_size,index_arr,algorithm_rank):
    col_column = ['1st','2nd','3rd']
    if algorithm_size <=3 :
        col_column = col_column[0:algorithm_size]
    else:
        algorithm_range = range(4,algorithm_size+1)
        for al in algorithm_range:
            col_column.append(str(al)+'th')
    ranksTable = pd.DataFrame(algorithm_rank,columns=col_column)
    ranksTable.index = index_arr
    return ranksTable
    
def bootstrap_analysis(Normalize_score_arr):
    length = len(Normalize_score_arr)
    metrics_columns=['accuracy', 'average_precision', 'f1', 'roc_auc','APaccuracy','BEP_score', 'RMSE','MXE']
    metrics_size = len(metrics_columns)
    algorithm_size = len(Normalize_score_arr[0])
    index_arr = np.array(Normalize_score_arr[0].index)
    frequency = 1000
    ranks = []
    fre_range = range(0,frequency)
    bootstrap_problem = np.random.randint(0, length, size=(frequency, length))
    bootstrap_metrics = np.random.randint(0, metrics_size, size=(frequency, metrics_size))
    for fre in fre_range:
        problem_arr = bootstrap_problem[fre]
        metrics_arr = bootstrap_metrics[fre]
        mean_performance = 0
        for problem in problem_arr:
            performance = 0
            Normalize_score = Normalize_score_arr[problem]
            for metrics in metrics_arr:
                column = metrics_columns[metrics]
                performance += Normalize_score[column]
            performance = performance/metrics_size
            mean_performance += performance
        mean_performance = mean_performance/length
        ranks.append(mean_performance.rank(ascending=False, method='first'))
    ranks_df = pd.concat(ranks,axis=1)
    algorithm_size_range = range(0,algorithm_size)
    algorithm_rank = []
    for al in algorithm_size_range:
        metrics_rank = []
        for mt in algorithm_size_range:
            row_list = list(ranks_df.iloc[al])
            count = row_list.count(mt+1)
            metrics_rank.append(float(count/frequency))
        algorithm_rank.append(metrics_rank)
        
    ranksTable =  rankTable(algorithm_size,index_arr,algorithm_rank)
    return ranks_df,ranksTable
    
#Performance_score
Adult_Normalize_score = pd.read_csv("E:\machine_learing\Adult_result\\Normalize_best_score.csv",index_col = 0)
Credit_Card_Normalize_score = pd.read_csv("E:\machine_learing\Credit_Card_result\\Normalize_best_score.csv",index_col = 0)
connect_Normalize_score = pd.read_csv("E:\machine_learing\connect-4_result\\Normalize_best_score.csv",index_col = 0)
Covtype_Normalize_score = pd.read_csv("E:\machine_learing\Covtype_result\\Normalize_best_score.csv",index_col = 0)
Eye_Normalize_score = pd.read_csv("E:\machine_learing\Eye_result\\Normalize_best_score.csv",index_col = 0)
Horse_Racing_Normalize_score = pd.read_csv("E:\machine_learing\Horse_Racing_result\\Normalize_best_score.csv",index_col = 0)
Magic_Normalize_score = pd.read_csv("E:\machine_learing\Magic_result\\Normalize_best_score.csv",index_col = 0)
Medical_Appointment_Normalize_score = pd.read_csv("E:\machine_learing\Medical_Appointment_result\\Normalize_best_score.csv",index_col = 0)
Occupancy_Normalize_score = pd.read_csv("E:\machine_learing\Occupancy_result\\Normalize_best_score.csv",index_col = 0)
shuttle_Normalize_score = pd.read_csv("E:\machine_learing\shuttle_result\\Normalize_best_score.csv",index_col = 0)

Normalize_best_score_arr = [Adult_Normalize_score,Credit_Card_Normalize_score,connect_Normalize_score,Covtype_Normalize_score,Eye_Normalize_score,Horse_Racing_Normalize_score,Magic_Normalize_score,Medical_Appointment_Normalize_score,Occupancy_Normalize_score,shuttle_Normalize_score]
Performance_score = getPerformance_score(Normalize_best_score_arr)
Performance_score.to_csv('E:\machine_learing\Performance_score.csv')

#Normalize_scoresbyProblem
Normalize_scoresbyProblem = pd.concat([Adult_Normalize_score['Mean'],Credit_Card_Normalize_score['Mean'],connect_Normalize_score['Mean'],Covtype_Normalize_score['Mean'],Eye_Normalize_score['Mean'],Horse_Racing_Normalize_score['Mean'],Magic_Normalize_score['Mean'],Medical_Appointment_Normalize_score['Mean'],Occupancy_Normalize_score['Mean'],shuttle_Normalize_score['Mean']],axis = 1)
Normalize_scoresbyProblem.columns= ['Adult', 'Credit_Card', 'connect','Covtype','Eye', 'Horse_Racing','Magic','Medical_Appointment', 'Occupancy','shuttle']
Normalize_scoresbyProblem['Mean'] = Normalize_scoresbyProblem.apply(lambda x: x.mean(),axis = 1)
Normalize_scoresbyProblem.to_csv('E:\machine_learing\\Normalize_scoresbyProblem.csv')

#OPT_score
Adult_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\Adult_result\\Normalize_OPT_score.csv",index_col = 0)
Credit_Card_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\Credit_Card_result\\Normalize_OPT_score.csv",index_col = 0)
connect_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\connect-4_result\\Normalize_OPT_score.csv",index_col = 0)
Covtype_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\Covtype_result\\Normalize_OPT_score.csv",index_col = 0)
Eye_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\Eye_result\\Normalize_OPT_score.csv",index_col = 0)
Horse_Racing_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\Horse_Racing_result\\Normalize_OPT_score.csv",index_col = 0)
Magic_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\Magic_result\\Normalize_OPT_score.csv",index_col = 0)
Medical_Appointment_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\Medical_Appointment_result\\Normalize_OPT_score.csv",index_col = 0)
Occupancy_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\Occupancy_result\\Normalize_OPT_score.csv",index_col = 0)
shuttle_Normalize_OPT_socre = pd.read_csv("E:\machine_learing\shuttle_result\\Normalize_OPT_score.csv",index_col = 0)
Normalize_OPT_socre_arr = [Adult_Normalize_OPT_socre,Credit_Card_Normalize_OPT_socre,connect_Normalize_OPT_socre,Covtype_Normalize_OPT_socre,Eye_Normalize_OPT_socre,Horse_Racing_Normalize_OPT_socre,Magic_Normalize_OPT_socre,Medical_Appointment_Normalize_OPT_socre,Occupancy_Normalize_OPT_socre,shuttle_Normalize_OPT_socre]
OPT_socre = getPerformance_score(Normalize_OPT_socre_arr)
OPT_socre.to_csv('E:\machine_learing\OPT_socre.csv')

#OPT-SEL
OPT_socre_mean = OPT_socre['Mean']
OPT_socre_mean.index = Performance_score.index
Performance_score['OPT-SEL'] = OPT_socre_mean

#Bootstrap 
ranks_df,ranksTable = bootstrap_analysis(Normalize_best_score_arr)
ranksTable.to_csv('E:\machine_learing\\ranksTable.csv')