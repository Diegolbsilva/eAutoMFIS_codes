from autoMFIS import *
import numpy as np
import scipy.io
from reweight import Reweight
from fuzzyfication import Fuzzification
from defuzzification import Defuzzification
import matplotlib.pyplot as plt

#Basic informations 
mat = scipy.io.loadmat('series/cluster1.mat')
data = mat.get('cluster1')
lag = 11
lag_notused = np.array([[4,5],[4,5],[4],[4,5]])
not_used_lag = False
h_prev = 18
#Actually, lag stands for all inputs for each serie. Example, lag = 2 uses s(t) and s(t-1) to predict s(t+1)
diff_series = True
detrend_series = False

#bin_values = 12; #Representação da binarização do tempo.
num_series = data.shape[1]  #Numero de series do problema, extraído dos dados

max_rulesize = 4 #Max numbers of premises rules.
min_activation = 0.20 #Minimum activation

#####Definicao de funcoes######
#detrend_method = ''
#bin_method = ''

fuzzy_method = 'mfdef_cluster'
num_groups = 5


total_number = data.shape[1]*lag

ensemble_rules = np.array([])
ensemble_antecedents = np.array([])
ensemble_prem_terms = np.array([])

#Concatenate rules
for i in range(5):
    not_select_subsample = np.random.choice(total_number,15)

    complete_rules, prem_terms, rules = autoMFIS(data,lag=lag, lag_notused=lag_notused, not_used_lag=not_used_lag, 
    not_select_subsample=not_select_subsample, h_prev = h_prev, diff_series=diff_series, detrend_series=detrend_series,
    num_series=num_series, max_rulesize=max_rulesize, min_activation=min_activation, fuzzy_method=fuzzy_method, num_groups=num_groups)

    #print(complete_rules)
    if ensemble_rules.size == 0:
        ensemble_rules = complete_rules
        ensemble_prem_terms = prem_terms
        ensemble_antecedents = rules
        print(ensemble_rules.shape)
    else:
        ensemble_rules = np.concatenate((ensemble_rules, complete_rules))
        ensemble_prem_terms = np.concatenate((ensemble_prem_terms,prem_terms))
        ensemble_antecedents = np.concatenate((ensemble_antecedents,rules))
        print(ensemble_rules.shape)
        print(ensemble_prem_terms.shape)
    print(ensemble_rules[:,0])




############NOW USING THE ENTIRE SET

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



rw = Reweight(mY_,complete_rules,prem_terms)
wd_, agg_training = rw.run('mqr',debug=False)


############## Defuzzification

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

    for n_rule in rules:
        #print('Rule {} is {}'.format(check_idx,test(n_rule,antecedents_activated)))
        if test(n_rule,antecedents_activated):
            rules_idx.append(check_idx)
        check_idx += 1
        
    prem_activated = np.zeros((ensemble_antecedents.shape[0],))
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
    
plt.show()

fig = plt.figure(figsize=(10,6))
for i in range(num_series):
    plt.subplot(num_series,1,i+1)
    plt.title('Serie {}'.format(i+1))
    plt.plot(yt_totest[:,i],color='blue')
    plt.plot(out_sample[:,i],color='red')
    plt.legend(['Predicted','Target'])
    
plt.show()
#plt.savefig('results/randomFIS_outsample.png')
plt.close()

