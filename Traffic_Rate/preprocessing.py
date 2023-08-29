import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
import random


class Preprocess():
    '''
    Preprocessing class for e-autoMFIS. 
    Preprocess has two main focus: apply some preprocess methods, such as diff series and detrend; and define lagged inputs for the model.
    Available methods:
    - diff_series
    - detrend_series
    - split_data
    - delay_input
    '''

    def __init__(self, data, lag, h_prev = 0, num_series = 0, target_position = -1):
        self.data = data
        self.h_prev = h_prev
        self.num_series = num_series
        self.target_position = target_position
        self.lag = lag


    def diff_series(self):
        '''
        Classical series differentiation.
        OUTPUT: In-sample data (in_sample), Out-sample data (out_sample)
        '''
        diff_data = self.data[1:,:] - self.data[0:self.data.shape[0]-1,:]
        in_sample = diff_data[:diff_data.shape[0]-self.h_prev,:]
        out_sample = diff_data[diff_data.shape[0]-self.h_prev:,:]

        return in_sample, out_sample

    def detrend_series(self):
        '''
        Detrend method using Linear Regression.
        OUTPUT: In-sample data (in_sample), Out-sample (out_sample) data and trends of each serie (trends)
        '''
            
        detrended = np.zeros((self.data.shape))
        trends = np.zeros((self.data.shape))

        for i in range(self.data.shape[1]):
            x_detrend = [k for k in range(0, self.data.shape[0])]
            x_detrend = np.reshape(x_detrend, (len(x_detrend), 1))
            y = self.data[:,i]
            model = LinearRegression()
            model.fit(x_detrend, y)
            # calculate trend
            trend = model.predict(x_detrend)
            trends[:,i] = [trend[k1] for k1 in range(0,len(x_detrend))]

            # detrend
            detrended[:,i] = [y[k2]-trend[k2] for k2 in range(0, len(x_detrend))]
            
            in_sample = detrended[:detrended.shape[0]-self.h_prev,:]
            out_sample = detrended[detrended.shape[0]-self.h_prev:,:]

        return in_sample, out_sample, trends

    def split_data(self, data=None):
        '''
        Split data into in-sample data and out-sample data
        OUTPUT: In-sample data (in_sample) and Out-sample data (out_sample)
        '''
        if data is None:
            in_sample = self.data[:self.data.shape[0]-self.h_prev,:]
            out_sample = self.data[self.data.shape[0]-self.h_prev:,:]
        else:
            in_sample = data[:data.shape[0]-self.h_prev,:]
            out_sample = data[data.shape[0]-self.h_prev:,:]

        return in_sample, out_sample

    
    def delay_input(self,in_sample=None, lag = 0):
        '''
        Prepare data for multivariate time series problem, creating delayed inputs. If in-sample data is not given, delay_input use
        the entire data instead.
        INPUT: in_sample (optional)
        OUTPUT: target (yt), non-delayed input (yp) and delayed-input (yp_lagged)
        '''
        if in_sample is not None:
            yt = np.zeros((in_sample.shape[0]-lag-1,self.num_series),dtype='float')
            yp = np.zeros((in_sample.shape[0]-lag-1,self.num_series), dtype='float')

            #Now delay inputs
            yp_lagged = np.zeros((in_sample.shape[0]-lag-1,self.num_series*lag),dtype='float')

            for i in range(self.num_series):
                yp[:,i] = in_sample[lag:in_sample.shape[0]-1,i]
                yt[:,i] = in_sample[lag+1:,i]
                for k in range(lag):
                    yp_lagged[:,i*lag+k] = in_sample[lag-k:in_sample.shape[0]-k-1,i]

        else:
            print('In-sample data not found. Using entire data instead')
            yt = np.zeros((self.data.shape[0]-self.h_prev-lag-1,self.num_series),dtype='float')

            #Todas as entradas defasadas 
            yp = np.zeros((self.data.shape[0]-self.h_prev-lag-1,self.num_series), dtype='float')
            yp_lagged = np.zeros((self.data.shape[0]-self.h_prev-lag-1,self.num_series*lag),dtype='float')

            for i in range(self.num_series):
                yp[:,i] = self.data[lag:self.data.shape[0]- self.h_prev - 1,i]
                yt[:,i] = self.data[lag+1:self.data.shape[0]- self.h_prev,i]
                for k in range(lag):
                    yp_lagged[:,i*lag+k] = self.data[lag-k:self.data.shape[0]- self.h_prev - k-1,i]

        return yt, yp, yp_lagged


    def spearman_corr_weights(self,in_sample=None):
        '''
        Using Spearman's correlation to create an array of probabilities to select subsamples (for linear correlation)
        OUTPUT: array of probabilities for each input
        '''
        if in_sample is None:
            in_sample, _ = self.split_data()

        spearman_corr_array = np.array([])
        for _input in range(0, in_sample.shape[1]):
            corr_value, _ = spearmanr(in_sample[:, _input], in_sample[:, self.target_position])
            spearman_corr_array = np.append(spearman_corr_array, abs(corr_value))

        probability_dist = spearman_corr_array/spearman_corr_array.sum()

        return probability_dist


    def mutual_information_corr_weights(self, in_sample=None):
        '''
        Using Mutual Information correlation technique to create an array of probabilities to select subsamples (for non-linear correlation)
        OUTPUT: array of probabilities for each input
        '''
        if in_sample is None:
            in_sample, _ = self.split_data()

        mi_score_array = np.array([])
        for _input in range(0, in_sample.shape[1]):
            mi_scores = mutual_info_score(in_sample[:, _input], in_sample[:, self.target_position])
            mi_score_array = np.append(mi_score_array, mi_scores)

        probability_dist = mi_score_array/mi_score_array.sum()

        return probability_dist
    
    
    def linear_acf_weights(self, in_sample=None):
        '''
        Using autocorrelation technique to create an array of probabilities to select the size of the subsamples selected (linear method). 
        OUTPUT: array of probabilities for each input
        '''
        if in_sample is None:
            in_sample, _ = self.split_data()
            
        n_rows, n_cols = in_sample.shape
        autocorrelation_matrix = np.zeros((self.lag, n_cols))
        for col in range(n_cols):
            column = in_sample[:, col]
            for lag in range(self.lag):
                if lag == 0:
                    autocorrelation_matrix[lag, col] = abs(np.corrcoef(column, column)[0, 1])
                else:
                    autocorrelation_matrix[lag, col] = abs(np.corrcoef(column[:-lag], column[lag:])[0, 1])

            autocorrelation_matrix[:, col] /= autocorrelation_matrix[:, col].sum()  
        return autocorrelation_matrix
    
    
    def non_linear_acf_weights(self, in_sample=None):
        '''
        Using autocorrelation technique to create an array of probabilities to select the size of the subsamples selected (linear method). 
        OUTPUT: array of probabilities for each input
        '''
        if in_sample is None:
            in_sample, _ = self.split_data()
            
        n_rows, n_cols = in_sample.shape
        autocorrelation_matrix = np.zeros((self.lag, n_cols))
        for col in range(n_cols):
            column = in_sample[:, col]
            for lag in range(self.lag):
                if lag == 0:
                    autocorrelation_matrix[lag, col] = abs(mutual_info_score(column, column))
                else:
                    autocorrelation_matrix[lag, col] = abs(mutual_info_score(column[:-lag], column[lag:]))

            autocorrelation_matrix[:, col] /= autocorrelation_matrix[:, col].sum()  
        return autocorrelation_matrix

    

    def random_sum_to(self, n, num_terms = None):
        num_terms = (num_terms or random.randint(2, n)) - 1
        a = random.sample(range(0, n), num_terms) + [0, n]
        list.sort(a)
        return [a[i+1] - a[i] for i in range(len(a) - 1)]

    
    def generate_subsamples(self, correlation_array, autocorrelation_matrix, num_inputs, in_sample = None, yt = None, yp = None, yp_lagged = None, vars_to_keep = None):
        
        if yt is None and yp is None and yp_lagged is None:
            yt, yp, yp_lagged = self.delay_input(in_sample, self.lag)
        
        if vars_to_keep is None:
            if in_sample.shape[1] <= num_inputs:
                vars_to_keep = np.random.choice(in_sample.shape[1], np.random.randint(1, in_sample.shape[1]+1), p=correlation_array, replace=False)
            else:
                vars_to_keep = np.random.choice(in_sample.shape[1], np.random.randint(1, num_inputs), p=correlation_array, replace=False)
        
        # yp[:, [i for i in range(in_sample.shape[1]) if i not in vars_to_keep]] = np.nan
        # yt[:, [i for i in range(in_sample.shape[1]) if i not in vars_to_keep]] = np.nan
        
        var_inputs = self.random_sum_to(num_inputs, len(vars_to_keep))
        corrected_lags = []
        for pos, _var in enumerate(vars_to_keep):
            lags = np.random.choice(range(self.lag), var_inputs[pos], p=autocorrelation_matrix[:, _var], replace=False)
            corrected_lags.append([(_var*self.lag) + (i) for i in lags])
            
        corrected_lags = [item for sublist in corrected_lags for item in sublist]
        yp_lagged[:, [i for i in range(in_sample.shape[1]*self.lag) if i not in corrected_lags]] = np.nan
        
        return yt, yp, yp_lagged