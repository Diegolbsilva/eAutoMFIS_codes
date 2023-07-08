import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from utils import *

import itertools
import json
import os

#For e-autoMFIS, we import all of them.
from autoMFIS import *
import numpy as np
import scipy.io
from reweight import Reweight
from fuzzyfication import Fuzzification
from defuzzification import Defuzzification
import matplotlib.pyplot as plt
from predict import predict



def convert_object_to_float(data,datanames):
    
    for name in datanames:
        dname = data[name].values.astype('str')
        dname = [new_value.replace(',','.') for new_value in dname]
        data[name] = dname
        data[name] = data[name].astype(float)
    return data

##Assertion for ensemble rules
#Somehow, when appending rules, ensemble_rules[:,1] has a erroneous form. This part seems to normalize it.

#This module correct this error. Also, we are going to use some assertion to check if rules contains the same antecedents.
def correct_bug(ensemble_rules,max_rulesize=0):
    correct_rule = []
    d_stacked_rules = []
    new_ensemble_rules = np.zeros(shape=ensemble_rules.shape, dtype=object)


    for  n_times in range(ensemble_rules.shape[1]):
        t_rules = ensemble_rules[:,n_times]
        correct_rule = []
        d_stacked_rules = []
        k = 0
        for rule in t_rules:
            #print(rule)
            #print(len(rule))
            #Check if there's a rule bigger than max_rulesize + 1 (#antecedents + #consequent)
            if len(rule) > max_rulesize + 1:
                #print(rule)
                for i in rule:
                    #k += 1
                    if isinstance(i,tuple):
                        #print(i)
                        correct_rule.append(i)
                    else:
                        if len(correct_rule) == 0:
                            #print(i)
                            pass
                            #d_stacked_rules.append(i)
                        else:
                            #print(correct_rule)
                            new_ensemble_rules[k,n_times] = correct_rule
                            k += 1
                            #d_stacked_rules.append(correct_rule)
                            correct_rule = []
                            #d_stacked_rules.append(i)
            else:
                new_ensemble_rules[k,n_times] = rule
                k += 1
                #d_stacked_rules.append(rule)
            
        #new_ensemble_rules[:,i] = np.array(d_stacked_rules)

    return new_ensemble_rules

def remove_duplicates(new_ensemble_rules,ensemble_prem_terms, ensemble_antecedents):
    t_rules = deepcopy(new_ensemble_rules[:,0])
    no_duplicated_ensemble = np.zeros(new_ensemble_rules.shape,dtype=object)
    no_duplicated_prem_terms = np.zeros(ensemble_prem_terms.shape)
    no_duplicated_antecedents = np.zeros(ensemble_antecedents.shape,dtype=object)
    new_t_rules = None

    k = 0
    j = 0
    for rule in t_rules:
        if new_t_rules is None:
            new_t_rules = [rule]
            no_duplicated_ensemble[k,:] = deepcopy(new_ensemble_rules[j,:])
            no_duplicated_prem_terms[k,:] = deepcopy(ensemble_prem_terms[j,:])
            no_duplicated_antecedents[k,:] = deepcopy(ensemble_antecedents[j,:])
            k += 1

        elif not check_duplicate_rules(rule,new_t_rules):
            new_t_rules.append(rule)
            no_duplicated_ensemble[k,:] = deepcopy(new_ensemble_rules[j,:])
            no_duplicated_prem_terms[k,:] = deepcopy(ensemble_prem_terms[j,:])
            no_duplicated_antecedents[k,:] = deepcopy(ensemble_antecedents[j,:])
            k += 1
        j += 1

    new_rules = deepcopy(no_duplicated_ensemble[:k,:])
    new_prem_terms = deepcopy(no_duplicated_prem_terms[:k,:])
    new_antecedents = deepcopy(no_duplicated_antecedents[:k,:])

    return new_rules, new_prem_terms, new_antecedents


old_data = pd.read_csv('series/AirQualityUCI.csv',sep=';',decimal='.')
old_data.dropna(thresh=1, inplace=True)
old_data.drop(labels=['Unnamed: 15', 'Unnamed: 16'],axis=1,inplace=True)
old_data.fillna(method='bfill',inplace=True)
inames = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
data = convert_object_to_float(old_data,inames)
numeric_data = data.select_dtypes(include=['float'])
ndata = numeric_data.where(numeric_data != -200)
ndata.fillna(method='bfill',inplace=True)
ndata.drop(labels=['NMHC(GT)','AH','RH'],axis=1,inplace=True)
dataframe = ndata[9000:]
dataframe.drop(labels=['PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)', 'T'],axis=1,inplace=True)
data = dataframe.values




#Basic informations 

num_predictors = 650
num_input = 7
lag = 26
lag_notused = np.array([[4,5],[4,5],[4],[4,5]])
not_used_lag = False
h_prev = 24
#Actually, lag stands for all inputs for each serie. Example, lag = 2 uses s(t) and s(t-1) to predict s(t+1)
diff_series = False
detrend_series = False

#bin_values = 12; #Representação da binarização do tempo.
num_series = data.shape[1]  #Numero de series do problema, extraído dos dados

max_rulesize = 4 #Max numbers of premises rules.
min_activation = 0.8 #Minimum activation

#####Definicao de funcoes######
#detrend_method = ''
#bin_method = ''

fuzzy_method = 'mfdef_cluster'
num_groups = 5

ensemble_rules = None

total_number = data.shape[1]*lag

if diff_series:
    diff_data = data[1:,:] - data[0:data.shape[0]-1,:]

    in_sample = diff_data[:diff_data.shape[0]-h_prev,:]
    
    out_sample = diff_data[diff_data.shape[0]-h_prev:,:]

elif detrend_series:
    detrended = np.zeros((data.shape))
    trends = np.zeros((data.shape))

    for i in range(data.shape[1]):
        x_detrend = [k for k in range(0, data.shape[0])]
        x_detrend = np.reshape(x_detrend, (len(x_detrend), 1))
        y = data[:,i]
        model = LinearRegression()
        model.fit(x_detrend, y)
        # calculate trend
        trend = model.predict(x_detrend)
        trends[:,i] = [trend[k1] for k1 in range(0,len(x_detrend))]
        # plot trend
        plt.plot(y)
        plt.plot(trend)
        plt.show()
        # detrend
        detrended[:,i] = [y[k2]-trend[k2] for k2 in range(0, len(x_detrend))]
        
    in_sample = detrended[:detrended.shape[0]-h_prev,:]
    out_sample = detrended[detrended.shape[0]-h_prev:,:]
    
else:
    in_sample = data[:data.shape[0]-h_prev,:]
    out_sample = data[data.shape[0]-h_prev:,:]


#Definicao do target
yt = np.zeros((in_sample.shape[0]-lag-1,num_series),dtype='float')

#Todas as entradas defasadas 
yp = np.zeros((in_sample.shape[0]-lag-1,num_series), dtype='float')
yp_lagged = np.zeros((in_sample.shape[0]-lag-1,num_series*lag),dtype='float')

for i in range(num_series):
    yp[:,i] = in_sample[lag:in_sample.shape[0]-1,i]
    yt[:,i] = in_sample[lag+1:,i]
    for k in range(lag):
        yp_lagged[:,i*lag+k] = in_sample[lag-k:in_sample.shape[0]-k-1,i]
        #print(i*lag+k)


###############Fuzzificacao

Fuzzyfy = Fuzzification(fuzzy_method)

#Lembrete: 
#axis 0 - Registros da série
#axis 1 - Valor de pertinência ao conjunto Fuzzy
#axis 2 - Numero de séries

first_time = True
for n in range(num_series):
    
    _, mf_params = Fuzzyfy.fuzzify(in_sample[:,n],np.array([]),num_groups=num_groups)
    mX, _ = Fuzzyfy.fuzzify(yp[:,n],mf_params,num_groups=num_groups)
    mY, _ = Fuzzyfy.fuzzify(yt[:,n],mf_params,num_groups=num_groups)
    if first_time:
        mX_ = np.ndarray([mX.shape[0],mX.shape[1], num_series])
        mY_ = np.ndarray([mY.shape[0],mY.shape[1], num_series])
        mf_params_ = np.ndarray([mf_params.shape[0],num_series])
        first_time = False
    mX_[:,:,n] = mX
    mY_[:,:,n] = mY
    mf_params_[:,n] = mf_params.ravel()
    #print(mf_params)
    #print(mX.shape)


mX_lagged_ = np.ndarray([mX_.shape[0],mX_.shape[1],yp_lagged.shape[1]])
for i in range(num_series):
    mf_params = mf_params_[:,i]
    for j in range(lag):
        mX, _ = Fuzzyfy.fuzzify(yp_lagged[:,i*lag+j],mf_params,num_groups=num_groups)
        mX_lagged_[:,:,i*lag+j] = mX
        #print(i*lag+j)


#mX_lagged_[:,:,not_select_subsample] = 0

#print(mX_lagged_[:,:,not_select_subsample])
############## Formulacao
if not_used_lag:
    new_mX, lags_used = remove_lags(mX_lagged_,lag_notused,num_series,lag)
else:
    new_mX = mX_lagged_


#Concatenate rules
for i in range(num_predictors):
    not_select_subsample = np.random.choice(total_number,total_number-num_input,replace=False)
    try:
        complete_rules, prem_terms, rules, agg_training, wd_ = autoMFIS(data,lag=lag, lag_notused=lag_notused, not_used_lag=not_used_lag,not_select_subsample=not_select_subsample, h_prev = h_prev, diff_series=diff_series, detrend_series=detrend_series, num_series=num_series, max_rulesize=max_rulesize, min_activation=min_activation, fuzzy_method=fuzzy_method, num_groups=num_groups)

        #Prediction of a single subset
        errors = predict(Fuzzyfy, lags_used = [], num_groups=num_groups, ndata=dataframe, data=data,in_sample=in_sample,out_sample=out_sample, lag = lag, mf_params_=mf_params_,num_series=num_series,agg_training=agg_training,yp_lagged=yp_lagged,h_prev=h_prev,not_used_lag=not_used_lag,n_attempt='subsample_{}'.format(i),wd_=wd_,ensemble_antecedents=rules,ensemble_rules=complete_rules)
        #print(errors)
        #print(complete_rules)
        if errors[0,0] < 1.0:
            if ensemble_rules is None:
                ensemble_rules = complete_rules
                ensemble_prem_terms = prem_terms
                ensemble_antecedents = rules
                #print(ensemble_rules.shape)
            else:
                ensemble_rules = np.concatenate((ensemble_rules, complete_rules))
                
                ensemble_prem_terms = np.concatenate((ensemble_prem_terms,prem_terms))
                ensemble_antecedents = np.concatenate((ensemble_antecedents,rules))
                print(ensemble_rules.shape)
                print(ensemble_prem_terms.shape)
            #print(ensemble_rules[:,0])
        else:
            print('RMSE Error {} greater than 1.0 for C02, ignoring'.format(errors[0,0]))
    except:
        pass

new_ensemble_rules = correct_bug(ensemble_rules,max_rulesize=max_rulesize)

new_rules, new_prem_terms, new_antecedents = remove_duplicates(new_ensemble_rules,ensemble_prem_terms, ensemble_antecedents)

rw = Reweight(mY_,new_rules,new_prem_terms)
wd_, agg_training = rw.run('mqr',debug=False)

predict(Fuzzyfy, lags_used = [], num_groups=num_groups, ndata=dataframe, data=data,in_sample=in_sample,out_sample=out_sample, lag = lag, mf_params_=mf_params_,num_series=num_series,agg_training=agg_training,yp_lagged=yp_lagged,h_prev=h_prev,not_used_lag=not_used_lag,n_attempt='_{}'.format(i),wd_=wd_,ensemble_antecedents=new_antecedents,ensemble_rules=new_rules)














































