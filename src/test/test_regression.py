import numpy as np 
import numpy.random as npr 
import pytest 

from regression import least_squares_fit

@pytest.fixture
def univariate_linear_problem(): 
    a, b = -5, 5
    params = (b - a) * npr.random(2) + a
    inputs = np.vstack((np.ones(20), np.linspace(-5, 5, 20))).T
    targets = inputs.dot(params)
    return inputs, targets, params

class TestLeastSquares: 
    def test_univariate_correctness(self, univariate_linear_problem): 
        inputs, targets, params = univariate_linear_problem
        assert np.allclose(least_squares_fit(inputs, targets), params)

class TestBasisRegression: 
    pass 

class TestBayesianRegression: 
    pass 

class TestNonlinearLeastSquares:
    pass 

class TestGaussianProcessRegression: 
    pass 