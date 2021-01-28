from os import makedirs, rmdir, remove
from os.path import exists, join
from pickle import dump, load
from shutil import rmtree

import numpy as np 
import numpy.random as npr 
from numpy.linalg import cholesky, norm 

from constants import cache

"""
Utilities for serializing and caching several datasets for quick prototyping
"""
 
def _setup_cache(overwrite: bool = False): 
    global cache
    if exists(cache) and overwrite is True: 
        rmtree(cache)
    elif exists(cache): 
        pass 
    else: 
        makedirs(cache) 

def _clean_or_load(path: str, overwrite: bool): 
    if exists(path) and overwrite: 
        remove(path) 
    elif exists(path): 
        with open(path, 'rb') as source: 
            return load(source)

def synthetic_regression(overwrite: bool = False, **kwargs) -> dict: 
    global cache 
    _setup_cache() 
    path = join(cache, "synthetic_regression.pkl")
    _clean_or_load(path, overwrite) 

    def _construct_covariance_matrix(x: np.ndarray, y: np.ndarray, covariance: callable) -> np.ndarray: 
        n, m = len(x), len(y) 
        k_xy = np.empty((n, m))
        for i in range(n): 
            for j in range(m): 
                k_xy[i, j] = covariance(x[i], y[j])
        return k_xy 

    inputs = kwargs.get("inputs", np.linspace(-3, 3, 25))
    covariance = kwargs.get("covariance", lambda x, y: np.exp(-norm(x-y) / kwargs.get("length_scale", 1)))
    mean = kwargs.get("mean", lambda x: 0 * x)

    covariance_matrix = _construct_covariance_matrix(inputs, inputs, covariance) 
    sqrt_covariance_matrix = cholesky(covariance_matrix) 
    targets = sqrt_covariance_matrix @ npr.randn((len(inputs))) + mean(inputs)[:, None]

    data_dict = dict(inputs=inputs, targets=targets, descr="synthetic regression dataset")
    with open(path, "wb") as destination:
        dump(data_dict, destination)

    return data_dict

def synthetic_classification(overwrite: bool = False, **kwargs) -> dict: 
    global cache
    _setup_cache()
    path = join(cache, "synthetic_classification.pkl")
    _clean_or_load(path, overwrite) 
    # TODO: sample from a mog or something? 
    raise NotImplementedError
