import numpy as np
from copy import deepcopy

'''
Script for fuzzy t-norm operation. Available options:
- Min - t-norm as minimum.
- Prod - t-norm as product.

INPUTS:
- values: numpy array containing the values.
- name: t-norm options (min or prod)

'''


def tnorm(values,name):
    v = deepcopy(values)
    if name is 'min':
        return tnorm_minimum(v)
    elif name is 'prod':
        return tnorm_product(v)

def tnorm_product(values):
    v = deepcopy(values)
    
    return np.prod(v,axis=0)

def tnorm_minimum(values):
    v = deepcopy(values)
    
    return v.min(axis=0)

if __name__ == "__main__":
    v = np.array([0.5, 0.3])
    print(f'Input values: {v}')
    print('Using min t-norm')
    print(tnorm(v,'min'))
    v2 = np.random.rand(2,2)
    print(5*'-')
    print(f'Input values: {v2}')
    print('Using product t-norm')
    print(tnorm(v2,'prod'))