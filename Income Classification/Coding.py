# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:25:09 2022

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import time 

data1 = pd.read_excel('C:/Users/nqhun/Downloads/adults.xlsx')
data = data1.copy()
# =============================================================================
# data.hist(figsize = (20,10))
# =============================================================================

data.head()
a= data.copy()
subset_data_catalog = a
b=subset_data_catalog.where(subset_data_catalog !=' ?')
for i in range(14):    
    b.iloc[:,i] = b.iloc[:,i].fillna(b.iloc[:,i].mode()[0])
data_use_ = b
data_use_
#%% =============================================================================

def find_prob_discrete_distribution(data,columns1):
    name_columns = data.columns
    count = pd.DataFrame(data.groupby(data[name_columns[columns1]])[name_columns[columns1]].count())
    prob = count/len(data.iloc[:,0])
    prob_df = pd.DataFrame(prob)
    prob_df['state'] = prob_df.index
    prob_df.index = range(len(prob_df.iloc[0:,]))
    return prob_df

def simulation_runif_acording_probilaty(n,columns,data):
    name_columns = data.columns
    count = pd.DataFrame(data.groupby(data[name_columns[columns]])[name_columns[columns]].count())
    state = list(count.index)
    state_prob = count.iloc[:,0]/len(data.iloc[:,0])
    sample = np.random.uniform(0,1,n)
    prob2 = [np.sum(state_prob[0:i]) for i in range(len(state_prob)+1)]
    state_simulation =[]
    for i in range(n):
        for j in range(len(state_prob)):
            if prob2[j]<sample[i]<=prob2[j+1]:
                statei = state[j]
                state_simulation.append(statei)
    return state_simulation

def simulation_runif_acording_probilaty_0(n,prob,state):
    state_prob = prob
    sample = np.random.uniform(0,1,n)
    prob2 = [np.sum(state_prob[0:i]) for i in range(len(state_prob)+1)]
    state_simulation =[]
    for i in range(n):
        for j in range(len(state_prob)):
            if prob2[j]<sample[i]<=prob2[j+1]:
                statei = state[j]
                state_simulation.append(statei)
    return state_simulation

def simulation_normal_distribution_discrete(data_income,num,columnsdata):
    name_columns = data_income.columns
    count = pd.DataFrame(data_income.groupby(data_income[name_columns[columnsdata]])[name_columns[columnsdata]].count())
    value1 = list(count.index)
    value3 = value1.copy()
    value2 = 0
    value3.insert(0,value2)
    mean = np.mean(data_income.iloc[:,columnsdata])
    std = np.std(data_income.iloc[:,columnsdata])
    prob_list=[ss.norm.cdf((value1[0]-mean)/std)]
    for i in range(len(value1)-1):
        prob = ss.norm.cdf((value1[i+1]-mean)/std)-ss.norm.cdf((value1[i]-mean)/std)
        prob_list.append(prob)
    result1 = simulation_runif_acording_probilaty_0(num, prob_list , value1)
    plt.hist(result1)
    return result1

def create_bins(lower_bound, width, quantity):
    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
    return bins

#%% funcion of bayer
def Kfold(data):
     data.index= range(len(data.index))
     index1 = np.random.choice(data.index,round(9/10*len(data.index)),replace=False)
     index2 = np.random.choice(data.index,round(1/10*len(data.index)),replace=False)
     data_train = data.iloc[index1,:]
     data_test = data.iloc[index2,:]
     return data_train,data_test
 

def prob_value_dis(data,column_num,value):
    list_value = data.iloc[:,column_num]
    count = data[(data.iloc[:,column_num]==value)].count()[0]
    p = count/data.shape[0]
    return p

def bayer_discrete(vector,data,list_columns):
    prob = 1
    for column in list_columns:
        prob = prob*prob_value_dis(data,column,vector[column])
    return prob

def bayer_gauus(vector1,data,list_columns):
    vector = list(vector1[list_columns])
    mean = np.mean(data.iloc[:,list_columns])
    std = np.std(data.iloc[:,list_columns])
    prob=1
    for num,columns in enumerate(list_columns):
        prob_value = ss.norm.pdf((vector[num]-mean[num])/std[num])
        prob = prob*prob_value
    return prob

def find_list_prob_discrete(data,list_columns):
    list_prob_of_columns = {}
    columns_name = data.columns
    for num,column in enumerate(list_columns):
        list_prob = [[],[]]
        count = pd.DataFrame(data.groupby(data[columns_name[column]])[columns_name[column]].count())
        groups = count.index
        prob = count/np.shape(data)[0]
        list_prob[0] = list(groups)
        list_prob[1] = list(prob.iloc[:,0])
        list_prob_of_columns[num] = list_prob
    return list_prob_of_columns

def bayer_discrete_with_prob(vector1, data,list_columns):
     vector = list(vector1[list_columns])
     list_prob_of_columns = find_list_prob_discrete(data,list_columns)
     prob = 1
     for num,columns in enumerate(list_columns):
         for i in range(len(list_prob_of_columns[0][0])):
             if vector[num] == list_prob_of_columns[0][0][i]:
                 p = list_prob_of_columns[0][1][i]
                 prob = prob*p
     return prob


def bayer_classified(data_test,data_train,list_columns_discret,list_columns_gause):
    predict = []
    list_prob_of_columns1 = find_list_prob_discrete(data_train[data_train['Lasary']==' <=50K'],list_columns_discret)
    list_prob_of_columns2 = find_list_prob_discrete(data_train[data_train['Lasary']==' >50K'],list_columns_discret)
    for i in range(np.shape(data_test)[0]):
        print(i)
        vector = data_test.iloc[i,:]
        vector
        prob1 = 1
        vector1 = list(vector[list_columns_discret])
        for num,columns in enumerate(list_columns_discret):
            for j1 in range(len(list_prob_of_columns1[num][0])):
                if vector1[num] == list_prob_of_columns1[num][0][j1]:
                    p1 = list_prob_of_columns1[num][1][j1]
                    prob1 = prob1*p1
        prob2 = 1
        for num,columns in enumerate(list_columns_discret):
            for j2 in range(len(list_prob_of_columns2[num][0])):
                if vector1[num] == list_prob_of_columns2[num][0][j2]:
                    p2 = list_prob_of_columns2[num][1][j2]
                    prob2 = prob2*p2
        prob_1 =bayer_gauus(vector, data_train[data_train['Lasary']==' <=50K'],list_columns_gause)*prob1
        prob_2 =bayer_gauus(vector, data_train[data_train['Lasary']==' >50K'],list_columns_gause)*prob2
        if prob_1>prob_2:
            predict.append(' <=50K')
        else:
            predict.append(' >50K')
    return predict

def under_sampling(data,column_respone):
    column_respone = 14
    data = data_use_
    name_columns = data.columns
    groups_list = pd.DataFrame(data.groupby([data.columns[column_respone]])[data.columns[column_respone]].count())
    list_group_name= groups_list.index
    max_group = groups_list.iloc[:,0].idxmax()
    min_group = groups_list.iloc[:,0].idxmin()
    count_value = min(groups_list.iloc[:,0])
    data_under_sampling = pd.DataFrame()
    data_under_sampling = data_under_sampling.append(data[data.iloc[:,column_respone]==min_group])
    for group in list_group_name:
        group = ' <=50K'
        if group != min_group:
            continue
        data_sampling = data[data.iloc[:,column_respone]==group]
        index = np.random.choice(data_sampling.index,count_value)
        data_under_sampling = data_under_sampling.append(data_sampling.iloc[index,:])
    return data_under_sampling

    

#%% =============================================================================
data_more_than50 = data_use_[data_use_['Lasary']==' >50K']
data_less_than50 = data_use_[data_use_['Lasary']==' <=50K']
#%% simulation
data_simulate = pd.DataFrame(columns=data_more_than50.columns)

for i in (1,3,5,6,7,8,9,13):
    data_simulate.iloc[:,i]=simulation_runif_acording_probilaty(30717-7650-7650,i, data_more_than50)

for i in (0,2,10,11,12):
    data_simulate.iloc[:,i] = simulation_normal_distribution_discrete(data_more_than50,30717-7650-7650,i)

fnlwgt = data_more_than50['fnlwgt']
std_fnlwgt = np.std(fnlwgt)
mean_fnlwgt = np.mean(fnlwgt)
data_simulate.iloc[:,2] = np.random.normal(mean_fnlwgt,std_fnlwgt,23067-7650).tolist()
data_used = data_use_.append(data_simulate)

data_test = pd.read_excel('C:/Users/nqhun/Downloads/test_file.xlsx')
data_test=data_test.where(data_test !=' ?')
data_test = data_test.dropna()
data_test
data_test['Lasary']=data_test['Lasary'].replace(' <=50K.',' <=50K')
data_test['Lasary']=data_test['Lasary'].replace(' >50K.',' >50K')
data_test[data_test['Lasary']==' <=50K']
data_used1 = data_used.copy()
data_used1['Lasary'] = data_used1['Lasary'].fillna(' >50K')

start = time.time()
predic1 = bayer_classified(data_test,data_used1,[1,3,5,6,7,8,9,13],[0,2,10,11,12])##predict
end = time.time()
end - start

data_test[data_test['Lasary']==' <=50K']
# funcion test acuracy
def test_acuracy(predict,data):
    n = np.shape(data)[0]
    t = np.shape(data[data['Lasary']==' <=50K'])[0]
    p = 0
    for i in range(t):
        if predict[i] == ' <=50K':
            p = p+1
    print(p)
    print(t - p)
    print(p/t)
    p1=0
    for i in range(t,n):
        if predict[i] == ' >50K':
            p1 = p1+1
    print(p1)
    print(n-t-p1)
    print(p1/(n-t))
    print((p1+p)/n)
    return (p1+p)/n



#%% Kfold
data_train1,data_test1 = Kfold(data_used)
data_train2,data_test2 = Kfold(data_used)
data_train3,data_test3 = Kfold(data_used)
data_train4,data_test4 = Kfold(data_used)
data_train5,data_test5 = Kfold(data_used)

predict1 = bayer_classified(data_test1, data_train1,[1,3,5,6,7,8,9,13],[0,2,10,11,12])
predict2 = bayer_classified(data_test2, data_train2,[1,3,5,6,7,8,9,13],[0,2,10,11,12])
predict3 = bayer_classified(data_test3, data_train3,[1,3,5,6,7,8,9,13],[0,2,10,11,12])
predict4 = bayer_classified(data_test4, data_train4,[1,3,5,6,7,8,9,13],[0,2,10,11,12])
predict5 = bayer_classified(data_test5, data_train5,[1,3,5,6,7,8,9,13],[0,2,10,11,12])

test_acuracy(predic1,data_test)
