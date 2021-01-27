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

def newton(function: callable, x_initial: np.ndarray) -> np.ndarray: 
    x_ = deepcopy(x_initial)
    derivative = grad(function) 
    for _ in range(kwargs.get("max_iterations", int(1e2))): 
        update_ = function(x_)/derivative(x_)
        x_ -= update_ 
        if norm(update) <= kwargs.get("convergence_tol", 1e-3): 
            break 
    return x_ 





    

def newton_rhapson(function: callable, x_0: np.ndarray, **kwargs) -> np.ndarray: 
    import pdb; pdb.set_trace()
    x_estimated = deepcopy(x_0)

    if x_0.size == 1: 
        derivative = grad(function) 
        _newton_update = lambda x: function(x)/derivative(x)
    elif x_0.size > 1: 
        jacobian = jacfwd(function) 

        if is_tall(jacobian) is True: 
            def _newton_update(x: np.ndarray) -> np.ndarray: 
                jacobian_psuedo_inverse = solve(jacobian(x) @ jacobian(x), jacobian(x).T)
                return jacobian_psuedo_inverse @ function(x) 

        else: 
            _newton_update = lambda x: solve(jacobian(x), -function(x)) + x 

    for _ in range(kwargs.get("maximum_interations", 100)): 
        _update = _newton_update(x_estimated) 
        x_estimated -= _update 
        if norm(_update) < kwargs.get("covergence_tol", 1e-3): 
            break 

    return x_estimated 

if __name__=="__main__": 

    f = lambda x: np.vstack((x, x))
    x_0 = np.array((3., 4., 5.), dtype=float)

    try: 
        newton_rhapson(f, x_0)
    except: 
        pass
