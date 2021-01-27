from pickle import dump 

import numpy as np 
import numpy.ranodm as npr 
from import numpy.linalg import chol, norm 

from constants import cache_config

cholesky = chol 

"""
Utilities for serializing and caching several datasets for quick prototyping
"""

"""
TODO: 
    * cache dir paths and names in a constants module 
    * setup serialization 
"""

def _setup_cache_dir(overwrite: bool = False): 
    # TODO: finish setting this up and testing it 
    raise NotImplementedError
    global cache_config
    root = cache_config["root"]

    # create a cache directory, or optionally overwrite an existing one 
    from os import makedirs, rmdir
    from os.path import exists, join
    from shutil import rmtree

    if exists(root) and overwrite: 
        rmtree(root)
        makedirs(root) 
    elif exists(root): 
        # TODO: logging or warning message, don't just fail silently 
        raise OSError

    makedirs(root); 

    # create the top level subdirectories 
    for dataset_type in cache_config["top_level"]: 
        makedirs(join(root, dataset_type))

    # regression (getting paths like this is redundant... rethink this) 
    regression_path = join(root, "regression") 
    dump(synthetic_regression(cache=True), regression_path) 

    # classification 
    classification_path = join(root, "classification") 
    dump()

def synthetic_regression(cache: bool = False, **kwargs): 

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
    
    if cache is True: 
        return dict(
                inputs=inputs, 
                targets=targets, 
                meta=dict(description="synthetic regression data sampled from a rbf gaussian process"))
    else: 
        return targets 

def synthetic_classification(cache: bool = True): 
    n_classes, n_per_class = map(kwargs.get, (("n_classes", 3), ("n_per_class", 25)))
    raise NotImplementedError
