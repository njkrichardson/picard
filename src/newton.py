from copy import deepcopy 

import jax.numpy as np 
from jax import grad, jacfwd
from numpy.linalg import solve, norm 

def newton_rhapson(function: callable, x_0: np.ndarray, **kwargs) -> np.ndarray: 
    x_estimated = deepcopy(x_0)

    if x_0.size == 1: 
        derivative = grad(function) 
        _newton_update = lambda x: function(x)/derivative(x)
    elif x_0.size > 1: 
        jacobian = jacfwd(function) 
        _newton_update = lambda x: solve(jacobian(x), -function(x)) + x 

    for _ in range(kwargs.get("maximum_interations", 100)): 
        _update = _newton_update(x_estimated) 
        x_estimated -= _update 
        if norm(_update) < kwargs.get("covergence_tol", 1e-3): 
            break 

    return x_estimated 
