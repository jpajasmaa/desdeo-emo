from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError
from desdeo_problem.testproblems.TestProblems import test_problem_builder
#from desdeo_problem import ExperimentalProblem
from desdeo_problem import DataProblem
from desdeo_emo.EAs.ProbRVEA import ProbRVEA
from pyDOE import lhs


import pandas as pd
import numpy as np


class SurrogateKriging(BaseRegressor):
    def __init__(self):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.m = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.reshape(-1, 1)

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.m = GaussianProcessRegressor(alpha=0, kernel=kernel, n_restarts_optimizer=9)
        self.m.fit(X, y)

    def predict(self, X):
        #y_mean, y_stdev = np.asarray(self.m.predict(X, return_std=True)).reshape(1,-1)
        y_mean, y_stdev = self.m.predict(X, return_std=True)
        y_mean = (y_mean.reshape(1,-1))
        y_stdev = (y_stdev.reshape(1,-1))
        return (y_mean, y_stdev)






if __name__=="__main__":

    problem_name = "ZDT1"
    prob = test_problem_builder(problem_name)

    x = lhs(30, 100)
    y = prob.evaluate(x)

    x_names = [f'x{i}' for i in range(1,31)]
    y_names = ["f1", "f2"]

    data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)

    #problem = ExperimentalProblem(data = datapd, objective_names=obj_names, variable_names=var_names, uncertainity_names=unc_names, evaluators = [obj_function1, obj_function2, obj_function3])
    problem.train(models=SurrogateKriging)
    evolver = ProbRVEA(
                problem, interact=False, n_iterations=10, n_gen_per_iter = 100,\
                     lattice_resolution=10, use_surrogates=True,  population_size=100, total_function_evaluations=150)#, number_of_update=u)
    #problem.train(models=GaussianProcessRegressor, model_parameters={'kernel': Matern(nu=1.5)})

    while evolver.continue_iteration():
        evolver.iterate()

    print("evolver", evolver.population.objectives[:5])
    print("done")


