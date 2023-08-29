
import pandas as pd 
import numpy as np
from math import sqrt
from copy import deepcopy
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_absolute_error, mean_squared_error
#For e-autoMFIS, we import all of them.
from eautoMFIS_V2 import autoMFIS
from fuzzyfication import Fuzzification
from preprocessing import Preprocess
from utils import *

from pymoo.core.problem import  ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
import sys

%load_ext memory_profiler

data = pd.read_csv('AirQualityUCI.csv',sep=';',decimal='.')

data.dropna(thresh=1, inplace=True)
data.drop(labels=['Unnamed: 15', 'Unnamed: 16'],axis=1,inplace=True)

data.fillna(method='bfill',inplace=True)

def convert_object_to_float(data,datanames):
    for name in datanames:
        dname = data[name].values.astype('str')
        dname = [new_value.replace(',','.') for new_value in dname]
        data[name] = dname
        data[name] = data[name].astype(float)
    return data

inames = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']

data = convert_object_to_float(data,inames)

data = data.select_dtypes(include=['float'])
   
data = data.where(data != -200)

# Preenchimento de dados faltantes
data.fillna(method='bfill',inplace=True)

# Preenchimento de dados faltantes
data.fillna(method='bfill',inplace=True)

data.drop(labels=['NMHC(GT)', 'PT08.S1(CO)','PT08.S2(NMHC)','PT08.S3(NOx)','PT08.S4(NO2)','PT08.S5(O3)'],axis=1,inplace=True)

data = data[9000:-24]
data = data[['NO2(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'CO(GT)']]
dataset = data.values

lag = 24
h_teste = 24
h_validation = 24
h_train = dataset.shape[0] - h_teste - h_validation

a = dataset.shape[0]
all_data = dataset[lag:,:]
training_data = dataset[lag:lag+h_train,:]
test_data = dataset[a - h_teste:a,:]

n_predictors = 60
num_input = 12
lag_notused = np.array([[4,5],[4,5],[4],[4,5]])
not_used_lag = False

#Actually, lag stands for all inputs for each serie. Example, lag = 2 uses s(t) and s(t-1) to predict s(t+1)
diff_series = False
detrend_series = False

num_series = training_data.shape[1]  #Numero de series do problema, extraído dos dados

# max_rulesize = 5 #Max numbers of premises rules.
# min_activation = 0.58 #Minimum activation

form_method = 'nmean'
split_method = 'FCD'
solve_method = 'mqr'
#####Definicao de funcoes######
#detrend_method = ''
#bin_method = ''

fuzzy_method = 'mfdef_cluster'
# num_groups = 7

defuzz_method = 'height'

ensemble_rules = None

total_number = training_data.shape[1]*lag


filepath = 'results V2'

target_position = 4

def fuzzy_external(fuzzy_method, num_series, training_set, num_groups, yp, yt, yp_lagged, lag):
    ###############Fuzzificacao

    Fuzzyfy = Fuzzification(fuzzy_method)

    #Lembrete: 
    #axis 0 - Registros da série
    #axis 1 - Valor de pertinência ao conjunto Fuzzy
    #axis 2 - Numero de séries

    first_time = True
    # print(f'Serie fuzzification')
    for n in range(num_series):
        _, mf_params = Fuzzyfy.fuzzify(training_set[:,n],np.array([]),num_groups=num_groups)
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

    # print('Creating for lag values')
    mX_lagged_ = np.ndarray([mX_.shape[0],mX_.shape[1],yp_lagged.shape[1]])
    for i in range(num_series):
        # print(f'Serie {i}')
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

    return Fuzzyfy, mX_, mY_, mf_params_, mX_lagged_

class Air_Quality_Problem(ElementwiseProblem):

    def __init__(self, **kwargs):
            
            vars = {
                # "lag": Integer(bounds=(16,24)),
                # "n_inputs": Integer(bounds=(10,20)),
                "max_rulesize": Integer(bounds=(5, 7)),
                "min_activation": Real(bounds=(0.5, 0.655)),
                # "form_method": Choice(options=["nmean", "freq", "mean"]),
                # "split_method": Choice(options=["FCD", "voting"]),
                # "solve_method": Choice(options=['mqr', 'None']),
                # "fuzzy_method": Choice(options=['mfdef_cluster', 'mfdef_triangle',  'mfdef_tukkey']),
                "num_groups": Choice(options=[3, 5, 7, 9]),
                # "defuzz_method": Choice(options=["height", "cog", "mom" ])
            }
            super().__init__(vars=vars, n_obj=3)


    def _evaluate(self, x,  out, *args, **kwargs):
        # cont = 0
        # if cont == 0:
        #     x = {'max_rulesize': 6, 'min_activation': 0.5709066532065377, 'num_groups': 9}
        #     cont += 1
        
        print(x)
        # lag = x["lag"]
        # num_input = x["n_inputs"]
        max_rulesize = x["max_rulesize"] #Max numbers of premises rules.
        min_activation = x["min_activation"] #Minimum activation
        # form_method = x["form_method"]
        # split_method = x["split_method"]
        # solve_method = x["solve_method"]
        # fuzzy_method = x["fuzzy_method"]
        num_groups = x["num_groups"]
        # defuzz_method = x["defuzz_method"]

        ensemble_rules = None

        # total_number = data_.shape[1]*lag

        filepath = 'results V2'
        
        target_position = 4

        preprocess_data = Preprocess(deepcopy(training_data), lag, h_prev=h_validation,num_series=num_series, target_position = target_position)

        training_val_set, test_set = preprocess_data.split_data()
        training_set, val_set = preprocess_data.split_data(data=training_val_set)
        correlation_array = preprocess_data.spearman_corr_weights(in_sample=training_set)
        autocorrelation_matrix = preprocess_data.linear_acf_weights(in_sample=training_set)

        yt, yp, yp_lagged = preprocess_data.delay_input(in_sample = training_set, lag = lag)
        all_yt, all_yp, all_lagged = preprocess_data.delay_input(in_sample = dataset, lag = lag)

        min_error = 400.0

        bres = min_error

        initial_values = all_lagged[yp_lagged.shape[0],:].reshape(1,-1)
        in_sample = deepcopy(training_set)
        out_sample = deepcopy(val_set)

        #Concatenate rules
        for i in range(n_predictors):
            try:
                _, _, yp_lagged_ = preprocess_data.generate_subsamples(correlation_array=deepcopy(correlation_array),
                                                                        autocorrelation_matrix=autocorrelation_matrix,
                                                                        num_inputs=num_input, 
                                                                        in_sample=in_sample, 
                                                                        yt = yt, 
                                                                        yp = yp, 
                                                                        yp_lagged = deepcopy(yp_lagged))
                print('='*30)
                
                Fuzzyfy, mX_, mY_, mf_params_, mX_lagged_ = fuzzy_external(fuzzy_method, 
                                                                            num_series, 
                                                                            training_set, 
                                                                            num_groups, 
                                                                            yp, 
                                                                            yt,
                                                                            deepcopy(yp_lagged_), 
                                                                            lag)
                print(f'mX_: {sys.getsizeof(mX_)}')
                print(f'mY_: {sys.getsizeof(mY_)}')
                print(f'mf_params_: {sys.getsizeof(mf_params_)}')
                print(f'mX_lagged_: {sys.getsizeof(mX_lagged_)}')
                
                model = autoMFIS(diff_series=diff_series,
                                detrend_series=detrend_series, 
                                fuzzy_method=fuzzy_method,
                                solve_method=solve_method,
                                defuzz_method=defuzz_method, 
                                num_groups = num_groups, 
                                inputs = num_input, 
                                h_prev = out_sample.shape[0],
                                num_series = num_series, 
                                max_rulesize = max_rulesize,
                                min_activation = min_activation, 
                                lag = lag, 
                                target_position = 4, 
                                hide_values = False, 
                                form_method = form_method, 
                                split_method = split_method, 
                                show=True)
                
                model.set_fuzzification(Fuzzyfy, mf_params_, mX_, mY_, deepcopy(mX_lagged_))

                t_mX_lagged, complete_rules, prem_terms, rules, agg_training, wd_ = model.train(data = dataset, 
                                                                                                correlation_array = correlation_array, 
                                                                                                autocorrelation_matrix = autocorrelation_matrix,
                                                                                                in_sample=in_sample, 
                                                                                                out_sample=out_sample, 
                                                                                                lag_notused=[],
                                                                                                debug=False)
                
                print(f't_mX_lagged: {sys.getsizeof(t_mX_lagged)}')
                print(f'agg_training: {sys.getsizeof(agg_training)}')

                predicted_values = np.zeros(test_set.shape)
                
                yt_totest, errors = model.predict(initial_values, 
                                                    data=dataset, 
                                                    in_sample = yt, 
                                                    out_sample=val_set,
                                                    agg_training=agg_training,
                                                    h_prev=h_teste,
                                                    n_attempt=f'p_subsample_{i}',
                                                    wd_=wd_,
                                                    ensemble_antecedents=rules,
                                                    ensemble_rules=complete_rules,
                                                    filepath=filepath,
                                                    lim=min_error, 
                                                    fig_axis=[4,2],
                                                    ndata=data.columns,
                                                    show=False,
                                                    plot_image = True)

                real_yt = deepcopy(yt_totest)

                # print(errors)
                res = errors[1,4]
                # print(res)
                if res < bres:
                    bres = res
                    if ensemble_rules is None:
                        ensemble_rules = complete_rules
                        ensemble_prem_terms = prem_terms
                        ensemble_antecedents = rules
                        # print(ensemble_rules.shape)
                    else:
                        ensemble_rules = np.concatenate((ensemble_rules, complete_rules))
                        
                        ensemble_prem_terms = np.concatenate((ensemble_prem_terms,prem_terms))
                        ensemble_antecedents = np.concatenate((ensemble_antecedents,rules))
                        # print(ensemble_rules.shape)
                        #print(ensemble_prem_terms.shape)
                    #print(ensemble_rules[:,0])
                elif ensemble_rules is None and i == n_predictors - 1:
                    ensemble_rules = complete_rules
                    ensemble_prem_terms = prem_terms
                    ensemble_antecedents = rules
                    # print('No rules match criteria. Using rules to fill the gap')
                
            except Exception as e:
                print(e)
                if 'shapes' in str(e) and i != 59:
                    del  yp_lagged_, Fuzzyfy, mX_, mY_, mf_params_, mX_lagged_, model 
                if 'values to unpack' in str(e) and i != 59:
                    del  yp_lagged_, Fuzzyfy, mX_, mY_, mf_params_, mX_lagged_
                if 'memory' in str(e) and i != 59:
                    del  yp_lagged_, Fuzzyfy, mX_, mY_, mf_params_, mX_lagged_, t_mX_lagged, complete_rules, prem_terms, rules, agg_training, wd_
                pass
            
        print(f'ensemble_rules: {sys.getsizeof(ensemble_rules)}')
        print(f'ensemble_prem_terms: {sys.getsizeof(ensemble_prem_terms)}')
        print(f'ensemble_antecedents: {sys.getsizeof(ensemble_antecedents)}')
        print('='*30)
        
        cnt = 0
        list_remove = []

        dict_val = {}

        for i in range(ensemble_prem_terms.shape[0]):
            except_one = np.copy(ensemble_prem_terms)
            v = except_one[i,:]
            idx = np.argwhere(v > 0.5).ravel()
            v = v[idx]
            rest = np.delete(except_one, i, axis=0)
            rest = rest[:,idx]
            cpare = np.tile(v,(rest.shape[0],1))
            m = np.minimum(rest,cpare) 
            M = np.maximum(rest,cpare) + 10e-15
            res = m/M
            mean = np.mean(res,axis=1)
            #plt.figure()
            #plt.hist(mean)
            
            vv = np.argwhere(mean > 0.6).ravel()

            if vv.shape[0] > 0:    
                vv[vv > i] += 1
                vv2 = np.append(vv,np.array([i]))

                eval_v = ensemble_prem_terms[vv2][:,idx]

                t = np.mean(eval_v,axis=1)

                keep_val = vv2[np.argmax(t)]
                vmax = np.max(t)
                if keep_val not in list_remove:
                    dict_val[keep_val] = 1
                    list_remove.append(keep_val)
                else:
                    dict_val[keep_val] = dict_val[keep_val] + 1

                cnt += np.argwhere(mean > 0.6).shape[0]

        filtered_rules = deepcopy(ensemble_rules[list_remove,:])
        filtered_prems = deepcopy(ensemble_prem_terms[list_remove,:])
        filtered_antecedents = deepcopy(ensemble_antecedents[list_remove,:])
        
        print(f'filtered_rules: {sys.getsizeof(filtered_rules)}')
        print(f'filtered_prems: {sys.getsizeof(filtered_prems)}')
        print(f'filtered_antecedents: {sys.getsizeof(filtered_antecedents)}')

        model.set_fuzzification(Fuzzyfy, mf_params_, mX_, mY_, mX_lagged_)
        filtered_wd_, filtered_agg_training = model.reweight_mf(mY_,filtered_rules,filtered_prems)
        print(f'filtered_wd_: {sys.getsizeof(filtered_wd_)}')
        print(f'filtered_agg_training: {sys.getsizeof(filtered_agg_training)}')

        
        try:
            _, filtered_errors = model.predict(initial_values, 
                                               data=dataset,
                                               out_sample=test_set, 
                                               agg_training=filtered_agg_training,
                                               h_prev=h_teste,
                                               n_attempt='filtered_model',
                                               wd_=filtered_wd_,
                                               ensemble_antecedents=filtered_antecedents,
                                               ensemble_rules=filtered_rules, 
                                               filepath=filepath, lim=min_error,
                                               fig_axis=[4,2],
                                               ndata=data.columns,
                                               show=False,
                                               plot_image = True) 

            out["F"] = [filtered_errors[1, target_position], filtered_rules.shape[0], filtered_rules.shape[1]]
            print(out)
            print(f'out: {sys.getsizeof(out)}')

            del training_val_set, training_set, val_set, test_set, correlation_array, autocorrelation_matrix, yt, yp, yp_lagged, all_yt, all_yp, all_lagged 
            del min_error, initial_values, in_sample, out_sample, yp_lagged_, Fuzzyfy, mX_, mY_, mf_params_, mX_lagged_, model, t_mX_lagged, complete_rules
            del prem_terms, rules, agg_training, wd_, predicted_values, yt_totest, errors, cnt, list_remove, dict_val,filtered_rules, filtered_prems 
            del filtered_antecedents, filtered_wd_, filtered_agg_training, filtered_errors
            
        except Exception as e:
            print(e)
            out["F"] = [np.inf, np.inf, np.inf]
            del training_val_set, training_set, val_set, test_set, correlation_array, autocorrelation_matrix, yt, yp, yp_lagged, all_yt, all_yp, all_lagged 
            del min_error, initial_values, in_sample, out_sample, yp_lagged_, Fuzzyfy, mX_, mY_, mf_params_, mX_lagged_, model, t_mX_lagged, complete_rules
            del prem_terms, rules, agg_training, wd_, predicted_values, yt_totest, errors, cnt, list_remove, dict_val,filtered_rules, filtered_prems 
            del filtered_antecedents, filtered_wd_, filtered_agg_training
            pass
        
algorithm = NSGA2(pop_size=10,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  eliminate_duplicates=MixedVariableDuplicateElimination(),
                  )

problem = Air_Quality_Problem()

res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               verbose=False)
