import numpy as np 

def least_squares_fit(inputs: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray: 
    return np.linalg.solve(inputs.T @ inputs + kwargs.get('ridge_penalty', 0) * np.eye(inputs.shape[1]), inputs.T.dot(targets))
