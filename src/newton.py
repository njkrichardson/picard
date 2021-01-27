from copy import deepcopy 

import jax.numpy as np 
from jax import grad, jacfwd
from numpy.linalg import solve, norm 

def is_tall(arr: np.ndarray) -> bool: 
    return arr.shape[0] > arr.shape[1]

def is_wide(arr: np.ndarray) -> bool: 
    return arr.shape[0] < arr.shape[1]

def is_square(arr: np.ndarray) -> bool: 
    return arr.shape[0] == arr.shape[1]

def newton(function: callable, x_initial: np.ndarray, **kwargs) -> np.ndarray: 
    x_ = deepcopy(x_initial)
    update = get_newton_update(function, x_.size)
    for _ in range(kwargs.get("max_iterations", int(1e2))): 
        print(x_)
        update_ = update(x_)
        x_ -= update_ 
        if norm(update_) <= kwargs.get("convergence_tol", 1e-3): 
            break 
    return x_ 

def get_newton_update(function: callable, output_dimension: int) -> callable: 
    if output_dimension > 1: 
        if is_tall(jacobian) is True: 
            def _newton_update(x: np.ndarray) -> np.ndarray: 
                jacobian_psuedo_inverse = solve(jacobian(x) @ jacobian(x), jacobian(x).T)
                return jacobian_psuedo_inverse @ function(x) 
            return _newton_update
        else: 
            return lambda x: solve(jacobian(x), -function(x)) + x 
    else: 
        return lambda x: function(x)/grad(function)(x)
