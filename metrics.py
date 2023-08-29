import numpy as np
from math import sqrt

#A = Actual; F = Forecast

def mape(A,F):
    return np.mean(np.abs(np.divide(A-F,A)))

def smape(A, F):
    return np.mean(2*abs(A-F) / (abs(A) + abs(F)))

def rrse(A,F):
    num = sqrt(np.sum((F-A)**2))
    den = sqrt(np.sum((A-np.mean(A))**2))
    return num/(den+0.0000001)