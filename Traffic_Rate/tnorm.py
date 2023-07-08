import numpy as np

def tnorm(values,name):
    if name is 'min':
        return tnorm_minimum(values)
    elif name is 'prod':
        return tnorm_product(values)

def tnorm_product(values):
    return np.prod(values,axis=0)

def tnorm_minimum(values):
    return values.min(axis=0)
