from collections import namedtuple
from functools import partial 
import pdb
from warnings import warn 
from typing import Union

import numpy as np 

def polynomial(x: Union[np.ndarray, float], degree: int) -> np.ndarray: 
    return np.array([np.power(x, i) for i in range(1, degree)])

def fourier_basis(x: Union[np.ndarray, float], freqs: np.ndarray = np.arange(1, 4)) -> np.ndarray: 
    arr = np.empty(freqs.size*2+1)
    pdb.set_trace()
    arr[0] = x
    for i, freq in zip(range(1, freqs.size*2-1, 2), freqs): 
        arr[i] = np.cos(freq*x)
        arr[i+1] = np.sin(freq*x)
    return arr 

def add_bias(arr: np.ndarray) -> np.ndarray: 
    return np.vstack((np.ones_like(arr[:, 0]), arr.T)).T

def construct_feature_matrix(inputs: np.ndarray, feature_map: callable = None, **kwargs)-> np.ndarray:
    if feature_map is None: 
        if inputs.ndim == 1: inputs = inputs[..., np.newaxis]
        return add_bias(inputs) if kwargs.get('add_bias', False) is True else inputs
    try: 
        n, _ = inputs.shape 
    except: 
        n = inputs.shape[0]
    if kwargs.get('image_dim', None) is not None:  
        k = np.empty((n, kwargs.get('image_dim')), dtype='float64')
        for i in range(n): 
            k[i] = feature_map(inputs[i])
    else: 
        k = []
        for i in range(n): 
            k.append(feature_map(inputs[i]))
        k = np.array(k, dtype='float64')
    return add_bias(k) if kwargs.get('add_bias', False) is True else k 

def construct_regularizer(inputs: np.ndarray, ridge: float=0, **kwargs): 
    _, feature_dim = inputs.shape 
    return ridge * np.eye(feature_dim)

def least_squares_fit(inputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray: 
    features = construct_feature_matrix(inputs, **kwargs)
    regularizer = construct_regularizer(features, **kwargs)
    return np.linalg.solve(features.T @ features + regularizer, features.T.dot(targets))

if __name__ == "__main__":
    inputs = np.arange(0, 5) 
    targets = 2 * inputs + 6 
    # p = 3
    # params = least_squares_fit(inputs, targets, add_bias=True)
    # print(f"Params: {params}")
    print(fourier_basis(3))

