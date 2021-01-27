from os import makedirs, rmdir, remove
from os.path import exists, join
from pickle import dump 
from shutil import rmtree
from warnings import warn 

import numpy as np 
import numpy.ranodm as npr 
from import numpy.linalg import chol, norm 

from constants import cache_config

cholesky = chol 

"""
Utilities for serializing and caching several datasets for quick prototyping
"""

def _clean_cache(cache: str): 
    rmtree(cache)
    makedirs(cache) 

def _setup_cache_dir(overwrite: bool = False): 
    global cache

    # create a cache directory, or optionally overwrite an existing one 
    if exists(cache) and overwrite: 
        _clean_cache(cache)
    elif exists(cache): 
        warn("tried to setup a cache directory but one already exists")
        raise OSError
    else: 
        makedirs(cache); 

def synthetic_regression(overwrite: bool = False, **kwargs) -> list: 
    path = join(cache, "synthetic_regression.npy")
    if exists(path) and overwrite is True: 
        remove(path)
    elif exists(path): 
        return np.load(path, allow_pickle=True)

    def _construct_covariance_matrix(x: np.ndarray, y: np.ndarray, covariance: callable) -> np.ndarray: 
        n, m = len(x), len(y) 
        k_xy = np.empty((n, m))
        for i in range(n): 
            for j in range(m): 
                k_xy[i, j] = covariance(x[i], y[j])
        return k_xy 

    inputs = kwargs.get("inputs", np.linspace(-3, 3, 25))
    covariance = kwargs.get("covariance", lambda x, y: np.exp(-norm(x-y) / kwargs.get("length_scale", 1)))
    mean = kwargs.get("mean", lambda x: 0)

    covariance_matrix = _construct_covariance_matrix(inputs, inputs, covariance) 
    sqrt_covariance_matrix, _  = cholesky(covariance_matrix) 
    targets = sqrt_covariance_matrix @ npr.randn(len(inputs)) + mean(inputs)[:, None]

    np.save(path, [inputs, targets], allow_pickle=True)

def synthetic_classification(): 
    raise NotImplementedError
