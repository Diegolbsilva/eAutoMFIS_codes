import numpy as np
import pandas as pd 

from fuzzyfication import Fuzzification
from tnorm import *
from formulation import Formulation
from split import Split
from reweight import Reweight
from defuzzification import Defuzzification

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from math import sqrt


from basicfuzzy import trimf, trapmf


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def hybrid_func(values,min_act):
    values = np.asarray(values)
    if np.sum(values) == 0:
        return False, 0, 0
    else:
        mean_activation = np.mean(values[values>0],axis=0)
        freq = np.sum(values>0)
        freq_rel = freq/len(values)

        check_activation = mean_activation*freq_rel
        if check_activation > min_act:
            activation = True
        else:
            activation = False
        return activation, mean_activation, check_activation
    
def freq_func(values,min_act):
    values = np.asarray(values)
    if np.sum(values) == 0:
        return False, 0, 0
    else:
        freq = np.sum(values>0)
        freq_rel = freq/len(values)

        check_activation = freq_rel
        if check_activation > min_act:
            activation = True
        else:
            activation = False
        return activation, freq_rel, check_activation

def card_func(values,min_act):
    values = np.asarray(values)
    if np.sum(values) == 0:
        return False, 0, 0
    else:
        mean_activation = np.mean(values[values>0],axis=0)

        check_activation = mean_activation
        if check_activation > min_act:
            activation = True
        else:
            activation = False
        return activation, mean_activation, check_activation


def remove_lags(mX_lagged_,lag_notused,num_series,lag):
    assert num_series == lag_notused.shape[0]
    lags_used = np.array(lag_notused)
    for n in range(num_series):
        lag_serie = lag_notused[n]
        lin = np.linspace(0,lag-1,lag)
        lag_used = np.setdiff1d(lin,lag_serie) + n*lag
        lag_used = [int(f) for f in lag_used]
        lags_used[n] = lag_used
        print(lag_used)
        if n == 0:
            new_mX = mX_lagged_[:,:,lag_used[:]]   
            #print(new_mX.shape) 
        else:
            new_mX = np.concatenate((new_mX,mX_lagged_[:,:,lag_used]),axis=2)
    
    return new_mX, lags_used

def mape(A,F):
    return np.mean(np.abs(np.divide(A-F,A)))

def find_rules_by_antecedent(rules,val):
    index_ = []
    k = 0
    for rule in rules:
        #print(rule)
        if check_if_inside(val,rule[0]):
            index_.append(k)
        k += 1
    #print(index_)
    return index_


def test(rule,antecedents_activated):
    for term in rule[0]:
        
        if term in antecedents_activated:
            pass
        else:
            return False
    
    return True

def prem_term(rule,muX):
    prem_concat = []
    for term in rule:
        #print(term)
        prem_concat.append(muX[0,term[1],term[0]])
    return tnorm_product(prem_concat)




def autoMFIS(data, lag, lag_notused, not_used_lag, not_select_subsample):


    h_prev = 18
                      #Actually, lag stands for all inputs for each serie. Example, lag = 2 uses s(t) and s(t-1) to predict s(t+1)
    diff_series = False
    detrend_series = False

    #bin_values = 12; #Representação da binarização do tempo.
    num_series = data.shape[1]  #Numero de series do problema, extraído dos dados

    max_rulesize = 4 #Max numbers of premises rules.
    min_activation = 0.15 #Minimum activation

    #####Definicao de funcoes######
    #detrend_method = ''
    #bin_method = ''

    fuzzy_method = 'mfdef_cluster'
    num_groups = 9


    ############### Ajustes dos dados

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


    mX_lagged_[:,:,not_select_subsample] = 0

    #print(mX_lagged_[:,:,not_select_subsample])
    ############## Formulacao
    if not_used_lag:
        new_mX, lags_used = remove_lags(mX_lagged_,lag_notused,num_series,lag)
    else:
        new_mX = mX_lagged_
    


    form = Formulation(max_rulesize,min_activation)

    rules, rulesize, prem_terms = form.run(new_mX)

    split = Split(mY_,prem_terms,num_series)

    complete_rules = split.run(rules)




    ############## Reweight

    rw = Reweight(mY_,complete_rules,prem_terms)
    wd_, agg_training = rw.run('mqr',debug=False)


    ############## Defuzzification

    defuzz = Defuzzification(mf_params_,num_series)

    y_predict_ = defuzz.run('cog',agg_training)



    for i in range(num_series):
        idx = np.where(np.isnan(y_predict_[:,i]))

        if len(idx) > 0:
            y_predict_[idx,i] = 0
            '''
            print('There are {} NaN in prediction'.format(len(idx[0])))
        print('MAE score for serie {} is {}'.format(i+1,mean_absolute_error(yt[:,i], y_predict_[:,i])))
        print('RMSE for serie {} is {}'.format(i+1,sqrt(mean_squared_error(yt[:,i], y_predict_[:,i]))))
        print('SMAPE for serie {} is {}'.format(i+1,smape(yt[:,i], y_predict_[:,i])))
        print('MAPE for serie {} is {}'.format(i+1,mape(yt[:,i], y_predict_[:,i])))
        print('R2 score for serie {} is {}'.format(i+1,r2_score(yt[:,i], y_predict_[:,i])))
        print('----------------------')

    fig = plt.figure(figsize=(10,6))
    for i in range(num_series):
        plt.subplot(num_series,1,i+1)
        plt.title('Serie {}'.format(i+1))
        plt.plot(y_predict_[:,i],color='blue')
        plt.plot(yt[:,i],color='red')
        plt.legend(['Predicted','Target'])
    '''

    if diff_series:
        y__ = y_predict_ + data[lag:in_sample.shape[0]-1,:]
        yt__ = yt + data[lag:in_sample.shape[0]-1,:]
        for i in range(num_series):

            compare_series = data[lag:in_sample.shape[0]-1,i]
            plt.figure(figsize=(10,6))

            plt.title('Serie {}'.format(i+1))

            plt.plot(y__[:,i],color='blue')

            plt.plot(yt__[:,i],color='red')

            plt.legend(['Predicted','Target'])
            '''
            print('MAE score for serie {} is {}'.format(i+1,mean_absolute_error(yt__[:,i], y__[:,i])))
            print('RMSE for serie {} is {}'.format(i+1,sqrt(mean_squared_error(yt__[:,i], y__[:,i]))))
            print('SMAPE for serie {} is {}'.format(i+1,smape(yt__[:,i], y__[:,i])))
            print('MAPE for serie {} is {}'.format(i+1,mape(yt__[:,i], y__[:,i])))
            print('R2 score for serie {} is {}'.format(i+1,r2_score(yt__[:,i], y__[:,i])))
            print('----------------------')
            '''
    elif detrend_series:
        y__ = y_predict_ + trends[lag:in_sample.shape[0]-1,:]
        for i in range(num_series):

            compare_series = data[lag+1:in_sample.shape[0],i]
            plt.figure(figsize=(10,6))

            plt.title('Serie {}'.format(i+1))

            plt.plot(y__[:,i],color='blue')

            #plt.plot(data[lag:in_sample.shape[0]-1,i],color='red')

            plt.plot(compare_series, color='red')
            plt.legend(['Predicted','Target'])
            '''
            print('MAE score for serie {} is {}'.format(i+1,mean_absolute_error(compare_series, y__[:,i])))
            print('RMSE for serie {} is {}'.format(i+1,sqrt(mean_squared_error(compare_series, y__[:,i]))))
            print('SMAPE for serie {} is {}'.format(i+1,smape(compare_series, y__[:,i])))
            print('MAPE for serie {} is {}'.format(i+1,mape(compare_series, y__[:,i])))
            print('R2 score for serie {} is {}'.format(i+1,r2_score(compare_series, y__[:,i])))
            print('----------------------')
            '''


    ########### Prediction
    #Arrumação dos dados para previsão.
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
        prem_terms_test = np.zeros((rules.shape[0],1))

        for n_rule in rules:
            #print('Rule {} is {}'.format(check_idx,test(n_rule,antecedents_activated)))
            if test(n_rule,antecedents_activated):
                rules_idx.append(check_idx)
            check_idx += 1
            
        prem_activated = np.zeros((rules.shape[0],))
        for i in rules_idx:
            prem_activated[i,] = prem_term(rules[i,0],mX_values_in)
        
        agg_test = np.zeros((wd_.shape))
        for i in range(num_series):
            for j in rules_idx:
                rule = complete_rules[j,i]
                consequent = rule[-1]
                agg_test[j,consequent[1],i] = prem_activated[j,]
                
                
        weight_agg = np.multiply(agg_test,wd_)
        weight_ = np.zeros((weight_agg.shape[1],weight_agg.shape[2]))

        for i in range(weight_.shape[1]):
            weight_[:,i] = weight_agg[:,:,i].max(axis=0)

        w_todefuzz = np.reshape(weight_,(1,weight_.shape[0],weight_.shape[1]))
        
        
        y_pred = defuzz.run('cog',w_todefuzz)
        
        yt_totest[h_p,:] = y_pred
        
        yp_totest = np.roll(yp_totest,1)
        
        for i in range(num_series):
            yp_totest[0,i*lag] = y_pred[0][i]


    for i in range(num_series):
        idx = np.where(np.isnan(yt_totest[:,i]))

        if len(idx) > 0:
            yt_totest[idx,i] = 0


    if diff_series:
        Y__ = yt_totest + data[in_sample.shape[0]:data.shape[0]-1,:]
        Yt__ = out_sample + data[in_sample.shape[0]:data.shape[0]-1,:]

    elif detrend_series:
        Y__ = yt_totest + trends[in_sample.shape[0]:,:]
        Yt__ = out_sample + trends[in_sample.shape[0]:,:]

    else:
        Y__ = yt_totest
        Yt__ = out_sample

    '''
    for i in range(num_series):
        print('Outsample RMSE for serie {} is {} \n'.format(i+1,sqrt(mean_squared_error(Yt__[:,i], Y__[:,i]))))
        print('Outsample SMAPE for serie {} is {} \n'.format(i+1,smape(Yt__[:,i],Y__[:,i])))

    fig = plt.figure(figsize=(10,6))
    for i in range(num_series):
        plt.subplot(num_series,1,i+1)
        plt.title('Serie {}'.format(i+1))
        plt.plot(Y__[:,i],color='blue')
        plt.plot(Yt__[:,i],color='red')
        plt.legend(['Predicted','Target'])
        
    plt.savefig('results/randomFIS_outsample.png')
    plt.close()
    '''

    return complete_rules, prem_terms

