import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from utils import *

from autoMFIS import *
import numpy as np
import scipy.io
from reweight import Reweight
from fuzzyfication import Fuzzification
from defuzzification import Defuzzification
import matplotlib.pyplot as plt

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


def predict(Fuzzyfy, lags_used = [], num_groups=5, ndata=[''], data=[],in_sample=[],out_sample=[], lag = 0, mf_params_=[],num_series=[],agg_training='',yp_lagged='',h_prev=0,n_attempt=0,wd_=[],ensemble_antecedents=[],ensemble_rules=[],not_used_lag = False):
    defuzz = Defuzzification(mf_params_,num_series)

    y_predict_ = defuzz.run('cog',agg_training)

    yp_totest = yp_lagged[yp_lagged.shape[0]-1:yp_lagged.shape[0],:]
    yt_totest = np.zeros((h_prev,num_series))

    for h_p in range(h_prev):
        mX_values_in = np.zeros((1,mf_params_.shape[0],yp_totest.shape[1]))
        antecedents_activated = []
        it = 0
        for i in range(num_series):
            mf_params = mf_params_[:,i]
            for j in range(lag):

                mX, _ = Fuzzyfy.fuzzify(np.array([yp_totest[0,i*lag+j]]),mf_params,num_groups=num_groups)
                mX_values_in[:,:,i*lag+j] = mX


                idx_nonzero = np.where(mX[0,:] > 0)
                idx_nonzero = idx_nonzero[0]

                if not_used_lag:
                    for k in range(idx_nonzero.shape[0]):
                        if j in lags_used[i]:
                            antecedents_activated.append((it,idx_nonzero[k]))
                        else:
                            pass
                    it += 1
                
                else:
                    for k in range(idx_nonzero.shape[0]):
                        antecedents_activated.append((i*lag+j,idx_nonzero[k]))

        if not_used_lag:
            mX_values_in, _ = remove_lags(mX_values_in,lag_notused,num_series,lag)


        check_idx = 0
        rules_idx = []
        prem_terms_test = np.zeros((ensemble_antecedents.shape[0],1))

        for n_rule in ensemble_antecedents:
            #print('Rule {} is {}'.format(check_idx,test(n_rule,antecedents_activated)))
            if test(n_rule,antecedents_activated):
                rules_idx.append(check_idx)
            check_idx += 1
            
        prem_activated = np.zeros((ensemble_antecedents.shape[0],))
        for i in rules_idx:
            prem_activated[i,] = prem_term(ensemble_antecedents[i,0],mX_values_in)
        
        agg_test = np.zeros((wd_.shape))
        for i in range(num_series):
            for j in rules_idx:
                rule = ensemble_rules[j,i]
                consequent = rule[-1]
                agg_test[j,consequent[1],i] = prem_activated[j,]
                
                
        weight_agg = np.multiply(agg_test,wd_)
        weight_ = np.zeros((weight_agg.shape[1],weight_agg.shape[2]))

        for i in range(weight_.shape[1]):
            weight_[:,i] = weight_agg[:,:,i].max(axis=0)

        w_todefuzz = np.reshape(weight_,(1,weight_.shape[0],weight_.shape[1]))
        
        
        y_pred = defuzz.run('cog',w_todefuzz,show=False)
        
        yt_totest[h_p,:] = y_pred
        
        y_temp = np.zeros(yp_totest.shape)
        assert y_temp.shape == yp_totest.shape
        y_temp[0,1:] = yp_totest[0,0:yp_totest.shape[1]-1]
        for ii in range(num_series):
            #print(yp_totest[0,ii*lag])
            #print(y_pred[0][ii])
            #yp_totest[0,ii*lag] = y_pred[0][ii]
            y_temp[0,ii*lag] = y_pred[0][ii]
            #print(yp_totest[0,yp_totest.shape[1]-1])
        yp_totest = y_temp

    plot_training(y_predict_=y_predict_,num_series=num_series,in_sample=in_sample,lag=lag,ndata=ndata,data=data,trends=[],filename='Insample {}'.format(n_attempt))

    plot_predict(yt_totest=yt_totest,num_series=num_series,data=data,in_sample=in_sample,out_sample=out_sample,trends=[],ndata=ndata,filename='Outsample {}'.format(n_attempt))

def plot_training(diff_series=False,detrend_series=False, y_predict_=[],num_series=0,in_sample=[],lag=0,ndata=[],data=[],trends=[],filename=[]):
    y_predict_new = np.ndarray(shape=y_predict_.shape)
    for i in range(num_series):
        y_predict__ = y_predict_[:,i]
        y_predict__[y_predict__ < 0.1] = np.nan
        
        idx = np.where(np.isnan(y_predict__))
        if len(idx) > 0:
            y_predict_new[:,i] = pd.DataFrame(y_predict__).fillna(method='bfill',limit=9000).values.ravel()
            print(i)
        else:
            y_predict_new[:,i] = y_predict_[:,i]

    if diff_series:
        Y__ = y_predict_new + data[lag:in_sample.shape[0]-1,:]
        Yt__ = yt + data[lag:in_sample.shape[0]-1,:]

    elif detrend_series:
        Y__ = y_predict_new + trends[lag:in_sample.shape[0]-1,:]
        Yt__ = yt + trends[lag:in_sample.shape[0]-1,:]

    else:
        Y__ = y_predict_new
        Yt__ = in_sample[lag+1:]


    plt.figure(figsize=(16*2,10*3))
    k = 1
    for i in range(num_series):
        plt.subplot(3,2,k)
        plt.title('Serie {}'.format(ndata[i]),fontsize=30)
        plt.plot(Y__[:,i],color='blue')
        plt.plot(Yt__[:,i],color='red')
        plt.legend(['Predicted','Target'])
        plt.xlabel('Time(h)',fontsize=15)
        plt.ylabel('Value',fontsize=15)
        k += 1
    plt.savefig('results/{}.png'.format(filename))
    #plt.show()
    plt.close()

def plot_predict(diff_series=False,detrend_series=False, yt_totest=[],num_series=0,data=[],in_sample=[],out_sample=[],trends=[],ndata=[],filename=''):
    #yt_totest = yt_totest[:9,:]
    #out_sample = out_sample[:9,:]

    for i in range(num_series):
        idx = np.where(np.isnan(yt_totest[:,i]))

        if len(idx) > 0:
            yt_totest[:,i] = pd.DataFrame(yt_totest[:,i]).fillna(method='bfill').values.ravel()

    if diff_series:
        #Y__ = yt_totest + data[in_sample.shape[0]:data.shape[0]-1,:]
        #Yt__ = out_sample + data[in_sample.shape[0]:data.shape[0]-1,:]
        y_pp = np.roll(yt_totest,1,axis=0)
        y_pp[0,:] = data[in_sample.shape[0],:]
        y_tt = np.roll(out_sample,1,axis=0)
        y_tt[0,:] = data[in_sample.shape[0],:]
        Y__ = yt_totest + y_pp
        Yt__ = out_sample + y_tt
        print('diff series')

    elif detrend_series:
        Y__ = yt_totest + trends[in_sample.shape[0]:,:]
        Yt__ = out_sample + trends[in_sample.shape[0]:,:]

    else:
        Y__ = yt_totest
        Yt__ = out_sample

    with open('results/{}.txt'.format(filename),'w') as f:
        for i in range(num_series):
            print('Outsample RMSE for serie {} is {} \n'.format(i+1,sqrt(mean_squared_error(Yt__[:,i], Y__[:,i]))), file=f)
            print('Outsample SMAPE for serie {} is {} \n'.format(i+1,smape(Yt__[:,i],Y__[:,i])),file=f)

    plt.figure(figsize=(16*3,10*2))
    k = 1
    for i in range(num_series):
        plt.subplot(3,2,k)
        plt.title('Serie {}'.format(ndata[i]),fontsize=30)
        plt.plot(Y__[:,i],color='blue')
        plt.plot(Yt__[:,i],color='red')
        plt.legend(['Predicted','Target'])
        plt.xlabel('Time(h)',fontsize=15)
        plt.ylabel('Value',fontsize=15)
        k += 1
    plt.savefig('results/{}.png'.format(filename))    #plt.show()
    plt.close()

#Basic informations 

def eautomfis(data, ndata = [''], lag = 10, lag_notused = np.array([[4,5],[4,5],[4],[4,5]]),not_used_lag = False, h_prev = 24,
diff_series = False, detrend_series = False, num_series = 5, max_rulesize = 4, 
min_activation = 0.18, fuzzy_method = 'mfdef_cluster', num_groups = 5, num_ensemble=15):

    total_number = data.shape[1]*lag

    ensemble_rules = None
    #ensemble_rules = np.array([])
    #ensemble_antecedents = np.array([])
    #ensemble_prem_terms = np.array([])


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
            #plt.plot(y)
            #plt.plot(trend)
            #plt.show()
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
    for i in range(num_ensemble):
        not_select_subsample = np.random.choice(total_number,total_number-15,replace=False)

        complete_rules, prem_terms, rules, agg_training, wd_ = autoMFIS(data,lag=lag, lag_notused=lag_notused, not_used_lag=not_used_lag,not_select_subsample=not_select_subsample, h_prev = h_prev, diff_series=diff_series, detrend_series=detrend_series, num_series=num_series, max_rulesize=max_rulesize, min_activation=min_activation, fuzzy_method=fuzzy_method, num_groups=num_groups)

        #Prediction of a single subset
        predict(Fuzzyfy, lags_used = [], num_groups=num_groups, ndata=ndata, data=data,in_sample=in_sample,out_sample=out_sample, lag = lag, mf_params_=mf_params_,num_series=num_series,agg_training=agg_training,yp_lagged=yp_lagged,h_prev=h_prev,not_used_lag=not_used_lag,n_attempt='subsample_{}'.format(i),wd_=wd_,ensemble_antecedents=rules,ensemble_rules=complete_rules)
        #print(complete_rules)
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

    
    new_ensemble_rules = correct_bug(ensemble_rules,max_rulesize=max_rulesize)

    new_rules, new_prem_terms, new_antecedents = remove_duplicates(new_ensemble_rules,ensemble_prem_terms, ensemble_antecedents)

    rw = Reweight(mY_,new_rules,new_prem_terms)
    wd_, agg_training = rw.run('mqr',debug=False)

    predict(Fuzzyfy, lags_used = [], num_groups=num_groups, ndata=ndata, data=data,in_sample=in_sample,out_sample=out_sample, lag = lag, mf_params_=mf_params_,num_series=num_series,agg_training=agg_training,yp_lagged=yp_lagged,h_prev=h_prev,not_used_lag=not_used_lag,n_attempt='_{}'.format(i),wd_=wd_,ensemble_antecedents=new_antecedents,ensemble_rules=new_rules)