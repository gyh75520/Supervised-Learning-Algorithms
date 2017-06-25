# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:36:54 2017

@author: Vne
"""

import pandas as pd #数据分析
import numpy as np #科学计算
import matplotlib.pyplot as plt
from Normalize_score import getNormalizedScoreforDsExp


def getDrawArray(a_Summary_score,b_Summary_score,c_Summary_score,a_BaseLine_score,b_BaseLine_score,c_BaseLine_score):
    index = a_Summary_score.index
    a_Summary_score.index = a_Summary_score.index+'_2000'
    b_Summary_score.index = b_Summary_score.index+'_4000'
    c_Summary_score.index = c_Summary_score.index+'_6500'
    Summary_score = pd.concat([a_Summary_score,b_Summary_score,c_Summary_score])
    BaseLine_score = (a_BaseLine_score+b_BaseLine_score+c_BaseLine_score)/3
    Normalized_Score = getNormalizedScoreforDsExp(Summary_score,BaseLine_score,3)
    a_Score = Normalized_Score[0]['Mean']
    b_Score = Normalized_Score[1]['Mean']
    c_Score = Normalized_Score[2]['Mean']
    a_Score.index = index
    b_Score.index = index
    c_Score.index = index
    a_Score = a_Score.sort_values(ascending=False)
    Algorithm_name =  np.array(a_Score.index)
    #length = len(a_Summary_score)
    i_range = range(0,6)
    result_arr = []
    for i in i_range:
        name = Algorithm_name[i]
        Algorithm_name[i] = Algorithm_name[i].replace('_best','')
        result_arr.append([a_Score[name],b_Score[name],c_Score[name]])
    return Algorithm_name[0:6],result_arr
    
#Dataset 6500
Adult_6500_Summary_score = pd.read_csv("E:\machine_learing\Adult_result\\Summary_best_score.csv",index_col = 0)
Credit_Card_6500_Summary_score = pd.read_csv("E:\machine_learing\Credit_Card_result\\Summary_best_score.csv",index_col = 0)
connect_6500_Summary_score = pd.read_csv("E:\machine_learing\connect-4_result\\Summary_best_score.csv",index_col = 0)
Covtype_6500_Summary_score = pd.read_csv("E:\machine_learing\Covtype_result\\Summary_best_score.csv",index_col = 0)
Eye_6500_Summary_score = pd.read_csv("E:\machine_learing\Eye_result\\Summary_best_score.csv",index_col = 0)
Horse_Racing_6500_Summary_score = pd.read_csv("E:\machine_learing\Horse_Racing_result\\Summary_best_score.csv",index_col = 0)
Magic_6500_Summary_score = pd.read_csv("E:\machine_learing\Magic_result\\Summary_best_score.csv",index_col = 0)
Medical_Appointment_6500_Summary_score = pd.read_csv("E:\machine_learing\Medical_Appointment_result\\Summary_best_score.csv",index_col = 0)
Occupancy_6500_Summary_score = pd.read_csv("E:\machine_learing\Occupancy_result\\Summary_best_score.csv",index_col = 0)
shuttle_6500_Summary_score = pd.read_csv("E:\machine_learing\shuttle_result\\Summary_best_score.csv",index_col = 0)

Adult_6500_BaseLine_score = pd.read_csv("E:\machine_learing\Adult_result\\BaseLine_score.csv",index_col = 0)
Credit_Card_6500_BaseLine_score = pd.read_csv("E:\machine_learing\Credit_Card_result\\BaseLine_score.csv",index_col = 0)
connect_6500_BaseLine_score = pd.read_csv("E:\machine_learing\connect-4_result\\BaseLine_score.csv",index_col = 0)
Covtype_6500_BaseLine_score = pd.read_csv("E:\machine_learing\Covtype_result\\BaseLine_score.csv",index_col = 0)
Eye_6500_BaseLine_score = pd.read_csv("E:\machine_learing\Eye_result\\BaseLine_score.csv",index_col = 0)
Horse_Racing_6500_BaseLine_score = pd.read_csv("E:\machine_learing\Horse_Racing_result\\BaseLine_score.csv",index_col = 0)
Magic_6500_BaseLine_score = pd.read_csv("E:\machine_learing\Magic_result\\BaseLine_score.csv",index_col = 0)
Medical_Appointment_6500_BaseLine_score = pd.read_csv("E:\machine_learing\Medical_Appointment_result\\BaseLine_score.csv",index_col = 0)
Occupancy_6500_BaseLine_score = pd.read_csv("E:\machine_learing\Occupancy_result\\BaseLine_score.csv",index_col = 0)
shuttle_6500_BaseLine_score = pd.read_csv("E:\machine_learing\shuttle_result\\BaseLine_score.csv",index_col = 0)

#Dataset 4000
Adult_4000_Summary_score = pd.read_csv("E:\machine_learing\Adult_result_4000\\Summary_best_score.csv",index_col = 0)
Credit_Card_4000_Summary_score = pd.read_csv("E:\machine_learing\Credit_Card_result_4000\\Summary_best_score.csv",index_col = 0)
connect_4000_Summary_score = pd.read_csv("E:\machine_learing\connect-4_result_4000\\Summary_best_score.csv",index_col = 0)
Covtype_4000_Summary_score = pd.read_csv("E:\machine_learing\Covtype_result_4000\\Summary_best_score.csv",index_col = 0)
Eye_4000_Summary_score = pd.read_csv("E:\machine_learing\Eye_result_4000\\Summary_best_score.csv",index_col = 0)
Horse_Racing_4000_Summary_score = pd.read_csv("E:\machine_learing\Horse_Racing_result_4000\\Summary_best_score.csv",index_col = 0)
Magic_4000_Summary_score = pd.read_csv("E:\machine_learing\Magic_result_4000\\Summary_best_score.csv",index_col = 0)
Medical_Appointment_4000_Summary_score = pd.read_csv("E:\machine_learing\Medical_Appointment_result_4000\\Summary_best_score.csv",index_col = 0)
Occupancy_4000_Summary_score = pd.read_csv("E:\machine_learing\Occupancy_result_4000\\Summary_best_score.csv",index_col = 0)
shuttle_4000_Summary_score = pd.read_csv("E:\machine_learing\shuttle_result_4000\\Summary_best_score.csv",index_col = 0)

Adult_4000_BaseLine_score = pd.read_csv("E:\machine_learing\Adult_result\\BaseLine_score.csv",index_col = 0)
Credit_Card_4000_BaseLine_score = pd.read_csv("E:\machine_learing\Credit_Card_result\\BaseLine_score.csv",index_col = 0)
connect_4000_BaseLine_score = pd.read_csv("E:\machine_learing\connect-4_result\\BaseLine_score.csv",index_col = 0)
Covtype_4000_BaseLine_score = pd.read_csv("E:\machine_learing\Covtype_result\\BaseLine_score.csv",index_col = 0)
Eye_4000_BaseLine_score = pd.read_csv("E:\machine_learing\Eye_result\\BaseLine_score.csv",index_col = 0)
Horse_Racing_4000_BaseLine_score = pd.read_csv("E:\machine_learing\Horse_Racing_result\\BaseLine_score.csv",index_col = 0)
Magic_4000_BaseLine_score = pd.read_csv("E:\machine_learing\Magic_result\\BaseLine_score.csv",index_col = 0)
Medical_Appointment_4000_BaseLine_score = pd.read_csv("E:\machine_learing\Medical_Appointment_result\\BaseLine_score.csv",index_col = 0)
Occupancy_4000_BaseLine_score = pd.read_csv("E:\machine_learing\Occupancy_result\\BaseLine_score.csv",index_col = 0)
shuttle_4000_BaseLine_score = pd.read_csv("E:\machine_learing\shuttle_result\\BaseLine_score.csv",index_col = 0)

#Dataset 2000
Adult_2000_Summary_score = pd.read_csv("E:\machine_learing\Adult_result_2000\\Summary_best_score.csv",index_col = 0)
Credit_Card_2000_Summary_score = pd.read_csv("E:\machine_learing\Credit_Card_result_2000\\Summary_best_score.csv",index_col = 0)
connect_2000_Summary_score = pd.read_csv("E:\machine_learing\connect-4_result_2000\\Summary_best_score.csv",index_col = 0)
Covtype_2000_Summary_score = pd.read_csv("E:\machine_learing\Covtype_result_2000\\Summary_best_score.csv",index_col = 0)
Eye_2000_Summary_score = pd.read_csv("E:\machine_learing\Eye_result_2000\\Summary_best_score.csv",index_col = 0)
Horse_Racing_2000_Summary_score = pd.read_csv("E:\machine_learing\Horse_Racing_result_2000\\Summary_best_score.csv",index_col = 0)
Magic_2000_Summary_score = pd.read_csv("E:\machine_learing\Magic_result_2000\\Summary_best_score.csv",index_col = 0)
Medical_Appointment_2000_Summary_score = pd.read_csv("E:\machine_learing\Medical_Appointment_result_2000\\Summary_best_score.csv",index_col = 0)
Occupancy_2000_Summary_score = pd.read_csv("E:\machine_learing\Occupancy_result_2000\\Summary_best_score.csv",index_col = 0)
shuttle_2000_Summary_score = pd.read_csv("E:\machine_learing\shuttle_result_2000\\Summary_best_score.csv",index_col = 0)

Adult_2000_BaseLine_score = pd.read_csv("E:\machine_learing\Adult_result\\BaseLine_score.csv",index_col = 0)
Credit_Card_2000_BaseLine_score = pd.read_csv("E:\machine_learing\Credit_Card_result\\BaseLine_score.csv",index_col = 0)
connect_2000_BaseLine_score = pd.read_csv("E:\machine_learing\connect-4_result\\BaseLine_score.csv",index_col = 0)
Covtype_2000_BaseLine_score = pd.read_csv("E:\machine_learing\Covtype_result\\BaseLine_score.csv",index_col = 0)
Eye_2000_BaseLine_score = pd.read_csv("E:\machine_learing\Eye_result\\BaseLine_score.csv",index_col = 0)
Horse_Racing_2000_BaseLine_score = pd.read_csv("E:\machine_learing\Horse_Racing_result\\BaseLine_score.csv",index_col = 0)
Magic_2000_BaseLine_score = pd.read_csv("E:\machine_learing\Magic_result\\BaseLine_score.csv",index_col = 0)
Medical_Appointment_2000_BaseLine_score = pd.read_csv("E:\machine_learing\Medical_Appointment_result\\BaseLine_score.csv",index_col = 0)
Occupancy_2000_BaseLine_score = pd.read_csv("E:\machine_learing\Occupancy_result\\BaseLine_score.csv",index_col = 0)
shuttle_2000_BaseLine_score = pd.read_csv("E:\machine_learing\shuttle_result\\BaseLine_score.csv",index_col = 0)

#getDrawArray
Adult_label,Adult_draw = getDrawArray(Adult_2000_Summary_score,Adult_4000_Summary_score,Adult_6500_Summary_score,Adult_2000_BaseLine_score,Adult_6500_BaseLine_score,Adult_4000_BaseLine_score)
Credit_Card_label,Credit_Card_draw = getDrawArray(Credit_Card_2000_Summary_score,Credit_Card_4000_Summary_score,Credit_Card_6500_Summary_score,Credit_Card_2000_BaseLine_score,Credit_Card_6500_BaseLine_score,Credit_Card_4000_BaseLine_score)  
connect_label,connect_draw = getDrawArray(connect_2000_Summary_score,connect_4000_Summary_score,connect_6500_Summary_score,connect_2000_BaseLine_score,connect_6500_BaseLine_score,connect_4000_BaseLine_score) 
Covtype_label,Covtype_draw = getDrawArray(Covtype_2000_Summary_score,Covtype_4000_Summary_score,Covtype_6500_Summary_score,Covtype_2000_BaseLine_score,Covtype_6500_BaseLine_score,Covtype_4000_BaseLine_score) 
Eye_label,Eye_draw = getDrawArray(Eye_2000_Summary_score,Eye_4000_Summary_score,Eye_6500_Summary_score,Eye_2000_BaseLine_score,Eye_6500_BaseLine_score,Eye_4000_BaseLine_score)
Horse_Racing_label,Horse_Racing_draw = getDrawArray(Horse_Racing_2000_Summary_score,Horse_Racing_4000_Summary_score,Horse_Racing_6500_Summary_score,Horse_Racing_2000_BaseLine_score,Horse_Racing_6500_BaseLine_score,Horse_Racing_4000_BaseLine_score)
Magic_label,Magic_draw = getDrawArray(Magic_2000_Summary_score,Magic_4000_Summary_score,Magic_6500_Summary_score,Magic_2000_BaseLine_score,Magic_6500_BaseLine_score,Magic_4000_BaseLine_score)
Medical_Appointment_label,Medical_Appointment_draw = getDrawArray(Medical_Appointment_2000_Summary_score,Medical_Appointment_4000_Summary_score,Medical_Appointment_6500_Summary_score,Medical_Appointment_2000_BaseLine_score,Medical_Appointment_6500_BaseLine_score,Medical_Appointment_4000_BaseLine_score)
Occupancy_label,Occupancy_draw = getDrawArray(Occupancy_2000_Summary_score,Occupancy_4000_Summary_score,Occupancy_6500_Summary_score,Occupancy_2000_BaseLine_score,Occupancy_6500_BaseLine_score,Occupancy_4000_BaseLine_score)
shuttle_label,shuttle_draw = getDrawArray(shuttle_2000_Summary_score,shuttle_4000_Summary_score,shuttle_6500_Summary_score,shuttle_2000_BaseLine_score,shuttle_6500_BaseLine_score,shuttle_4000_BaseLine_score)


plt_x = [2000,4000,6500]
#绘图
import numpy 
length = len(Adult_draw)
a_range = range(0,length)

'''
#p1 = plt.subplot(121)
#p2 = plt.subplot(122)
plt.figure(figsize=(8,8),dpi=600)
for a in a_range:
    plt.plot([1,2,3],Adult_draw[a],label = Adult_label[a])
#plt.sca(p1)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.86,0.93,20,endpoint=True))
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0))
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Adult",fontsize=18)

plt.figure(figsize=(8,8),dpi=600)
for a in a_range:
    plt.plot([1,2,3],Credit_Card_draw[a],label = Credit_Card_label[a])
#plt.sca(p1)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.78,0.96,20,endpoint=True))
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0))
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Credit_Card",fontsize=18)
#p1.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))

plt.figure(figsize=(8,8),dpi=600)
for a in a_range:
    plt.plot([1,2,3],connect_draw[a],label = connect_label[a])
#plt.sca(p2)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.88,0.96,20,endpoint=True)) 
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0))
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Connect",fontsize=18)

plt.figure(figsize=(8,8),dpi=600)
for a in a_range:
    plt.plot([1,2,3],Covtype_draw[a],label = Covtype_label[a])
#plt.sca(p2)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.77,1,20,endpoint=True)) 
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0))
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Covtype",fontsize=18)
'''
'''
plt.figure(figsize=(12,12),dpi=600)
ax1 = plt.subplot(221) # 在图表2中创建子图1
ax2 = plt.subplot(222) # 在图表2中创建子图2
ax3 = plt.subplot(223) # 在图表2中创建子图1
ax4 = plt.subplot(224) # 在图表2中创建子图2

plt.sca(ax1)
for a in a_range:
    plt.plot([1,2,3],Adult_draw[a],label = Adult_label[a])
#plt.sca(p1)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.86,0.93,20,endpoint=True))
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Adult",fontsize=18)

plt.sca(ax2)
for a in a_range:
    plt.plot([1,2,3],Credit_Card_draw[a],label = Credit_Card_label[a])
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.78,0.96,20,endpoint=True))
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Credit_Card",fontsize=18)

plt.sca(ax3)
for a in a_range:
    plt.plot([1,2,3],connect_draw[a],label = connect_label[a])
#plt.sca(p2)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.88,0.96,20,endpoint=True)) 
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Connect",fontsize=18)

plt.sca(ax4)
for a in a_range:
    plt.plot([1,2,3],Covtype_draw[a],label = Covtype_label[a])
#plt.sca(p2)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.76,1,20,endpoint=True)) 
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Covtype",fontsize=18)

plt.savefig("E:\machine_learing\\draw1.jpg",dpi=400)




plt.figure(figsize=(13,18),dpi=600)
ax1 = plt.subplot(321) # 在图表2中创建子图1
ax2 = plt.subplot(322) # 在图表2中创建子图2
ax3 = plt.subplot(323) # 在图表2中创建子图1
ax4 = plt.subplot(324) # 在图表2中创建子图2
ax5 = plt.subplot(325) # 在图表2中创建子图1
ax6 = plt.subplot(326) # 在图表2中创建子图2
plt.sca(ax1)
for a in a_range:
    plt.plot([1,2,3],Eye_draw[a],label = Eye_label[a])
#plt.sca(p1)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.72,1,20,endpoint=True))
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Eye",fontsize=18)

plt.sca(ax2)
for a in a_range:
    plt.plot([1,2,3],Horse_Racing_draw[a],label = Horse_Racing_label[a])
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.985,1,20,endpoint=True))
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Horse_Racing",fontsize=18)

plt.sca(ax3)
for a in a_range:
    plt.plot([1,2,3],Magic_draw[a],label = Magic_label[a])
#plt.sca(p2)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.90,1,20,endpoint=True)) 
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Magic",fontsize=18)

plt.sca(ax4)
for a in a_range:
    plt.plot([1,2,3],Medical_Appointment_draw[a],label = Medical_Appointment_label[a])
#plt.sca(p2)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.75,0.92,20,endpoint=True)) 
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Medical_Appointment",fontsize=18)


plt.sca(ax5)
for a in a_range:
    plt.plot([1,2,3],Occupancy_draw[a],label = Occupancy_label[a])
#plt.sca(p1)
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.97,1,20,endpoint=True))
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Occupancy",fontsize=18)

plt.sca(ax6)
for a in a_range:
    plt.plot([1,2,3],shuttle_draw[a],label = shuttle_label[a])
plt.xticks([1,2,3], plt_x, rotation=0)
plt.yticks(numpy.linspace(0.995,1,20,endpoint=True))
plt.grid()
plt.legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8)
plt.xlabel("TrainingSet size")
plt.ylabel("Normalized Score")
plt.title("Shuttle",fontsize=18)
plt.savefig("E:\machine_learing\\draw3.jpg",dpi=400)
'''

def drawColumnar(ax,draw,label,title):
    plt.sca(ax)
    draw_2000 = []
    for a in a_range:
        draw_2000.append(draw[a][0]);
    draw_4000 = []
    for a in a_range:
        draw_4000.append(draw[a][1]);
    draw_6500 = []
    for a in a_range:
        draw_6500.append(draw[a][2]);
    #plt.figure(figsize=(8,6),dpi=600)
    bar_width = 0.25 
    opacity = 0.6
    index = np.arange(length)
    plt.bar(index+ 0.5*bar_width, draw_2000, bar_width,alpha=opacity, color='royalblue',label='TrainSet 2000')  
    plt.bar(index + 1.5*bar_width, draw_4000, bar_width,alpha=opacity,color='r',label='TrainSet 4000')
    plt.bar(index + 2.5*bar_width, draw_6500, bar_width,alpha=opacity,color='yellow',label='TrainSet 6500') 
    plt.ylabel('Normalized Score')  
    plt.title(title,fontsize=16)  
    #plt.grid()
    plt.xticks(index + 2*bar_width, label,rotation=20,fontsize=8)
    plt.ylim(0.6,1)
    #plt.yticks(numpy.linspace(0,1,4,endpoint=True))
    plt.legend(loc=4)
    for x,y in zip(index,draw_2000):
        plt.text(x+0.25, y+0.01, '%.3f' % y, ha='center', va= 'bottom',fontsize=5.5)
    for x,y in zip(index,draw_4000):
        plt.text(x+0.5, y+0.01, '%.3f' % y, ha='center', va= 'bottom',fontsize=5.5)
    for x,y in zip(index,draw_6500):
        plt.text(x+0.75, y+0.01, '%.3f' % y, ha='center', va= 'bottom',fontsize=5.5)

'''      
plt.figure(figsize=(12,12),dpi=600)
ax1 = plt.subplot(221) # 在图表中创建子图1
ax2 = plt.subplot(222) # 在图表中创建子图2
ax3 = plt.subplot(223) # 在图表中创建子图3
ax4 = plt.subplot(224) # 在图表中创建子图4

drawColumnar(ax1,Adult_draw,Adult_label,'Adult')
drawColumnar(ax2,Credit_Card_draw,Credit_Card_label,'Credit_Card')
drawColumnar(ax3,connect_draw,connect_label,'Connect')
drawColumnar(ax4,Covtype_draw,Covtype_label,'Covtype')
plt.savefig("E:\machine_learing\\drawColumnar1.jpg",dpi=400)


plt.figure(figsize=(12,18),dpi=600)
ax1 = plt.subplot(321) # 在图表2中创建子图1
ax2 = plt.subplot(322) # 在图表2中创建子图2
ax3 = plt.subplot(323) # 在图表2中创建子图1
ax4 = plt.subplot(324) # 在图表2中创建子图2
ax5 = plt.subplot(325) # 在图表2中创建子图1
ax6 = plt.subplot(326) # 在图表2中创建子图2

drawColumnar(ax1,Eye_draw,Eye_label,'Eye')
drawColumnar(ax2,Horse_Racing_draw,Horse_Racing_label,'Horse_Racing')
drawColumnar(ax3,Magic_draw,Magic_label,'Magic')
drawColumnar(ax4,Medical_Appointment_draw,Medical_Appointment_label,'Medical_Appointment')
drawColumnar(ax5,Occupancy_draw,Occupancy_label,'Occupancy')
drawColumnar(ax6,shuttle_draw,shuttle_label,'Shuttle')
plt.savefig("E:\machine_learing\\drawColumnar2.jpg",dpi=400)


'''
plt.figure(figsize=(30,12),dpi=600)
ax1 = plt.subplot(251) # 在图表中创建子图1
ax2 = plt.subplot(252) # 在图表中创建子图2
ax3 = plt.subplot(253) # 在图表中创建子图3
ax4 = plt.subplot(254) # 在图表中创建子图4
ax5 = plt.subplot(255) # 在图表2中创建子图1
ax6 = plt.subplot(256) # 在图表2中创建子图2
ax7 = plt.subplot(257) # 在图表2中创建子图1
ax8 = plt.subplot(258) # 在图表2中创建子图2
ax9 = plt.subplot(259) # 在图表2中创建子图1
ax10 = plt.subplot(2,5,10) # 在图表2中创建子图2

drawColumnar(ax1,Adult_draw,Adult_label,'Adult')
drawColumnar(ax2,Credit_Card_draw,Credit_Card_label,'Credit_Card')
drawColumnar(ax3,connect_draw,connect_label,'Connect')
drawColumnar(ax4,Covtype_draw,Covtype_label,'Covtype')
drawColumnar(ax5,Eye_draw,Eye_label,'Eye')
drawColumnar(ax6,Horse_Racing_draw,Horse_Racing_label,'Horse_Racing')
drawColumnar(ax7,Magic_draw,Magic_label,'Magic')
drawColumnar(ax8,Medical_Appointment_draw,Medical_Appointment_label,'Medical_Appointment')
drawColumnar(ax9,Occupancy_draw,Occupancy_label,'Occupancy')
drawColumnar(ax10,shuttle_draw,shuttle_label,'Shuttle')
plt.savefig("E:\machine_learing\\drawColumnar3.jpg",dpi=400)