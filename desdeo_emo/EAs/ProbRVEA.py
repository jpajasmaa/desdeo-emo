from typing import Dict, Union

from desdeo_emo.EAs.BaseEA import BaseDecompositionEA, eaError
from desdeo_emo.population.Population import Population

from desdeo_emo.selection.APD_Select_constraints import APD_Select
from desdeo_emo.selection.Prob_APD_Select import Prob_APD_select, Prob_APD_select_v3  # superfast y considering mean APD
import numpy as np
from desdeo_tools.interaction import (
    SimplePlotRequest,
    ReferencePointPreference,
    validate_ref_point_data_type,
    validate_ref_point_dimensions,
    validate_ref_point_with_ideal,
)
from desdeo_problem import DataProblem, MOProblem
from numpy.core.numeric import indices
from numpy.lib.arraysetops import unique
from pandas.core.frame import DataFrame
from desdeo_tools.scalarization.ASF import SimpleASF, ReferencePointASF
from numba import njit
import numpy as np
import pandas as pd
import copy
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct,\
    WhiteKernel, RBF, Matern, ConstantKernel
from desdeo_problem import MOProblem
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from desdeo_problem import ExperimentalProblem
import sys 
import numpy as np
import pandas as pd
from desdeo_tools.utilities import fast_non_dominated_sort, hypervolume_indicator

from sklearn.cluster import KMeans

from sklearn.gaussian_process.kernels import Matern
from pymoo.factory import get_problem, get_reference_directions
import copy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError


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


#TODO: add @njit function here
def remove_duplicate(
    X: np.ndarray,
    archive_x: np.ndarray
    ):
    """identifiesthe duplicate rows for decision variables
    Args:
    X (np.ndarray): the current decision variables.
    archive_x (np.ndarray): The decision variables in the archive.
    Returns: 
    indicies (np.ndarray): the indicies of solutions that are NOT already in the archive.
    """

    all_variables = np.vstack((archive_x,X))
    all_variables_indicies = np.arange(len(all_variables))

    e,unique_variables_indicies = np.unique(all_variables, return_index=True, axis=0)

    repeated_all_variables = np.delete(all_variables_indicies, unique_variables_indicies)
    repeated_in_X = repeated_all_variables - len(archive_x)
    X_indicies = np.arange(len(X))
    X_uniqe_indicies = np.delete(X_indicies, repeated_in_X)
        
    return X_uniqe_indicies


# this right now maybe does not work properly..
# seems like doing stuff here only effects the problem archive.
# which is prob not the same as probrvea archive.
# yea i think so.
# so still need to combine the archives or no point using this. Disabled it for now.
def rvea_mm(
    #vectors: ReferenceVectors,
    individuals: np.ndarray,
    objectives: np.ndarray,
    uncertainity: np.ndarray,
    problem: MOProblem,
    #u: int
    ) -> float:
    """ Selects the solutions that need to be reevaluated with the original functions.
    This model management is based on the following papaer: 
    'P. Aghaei Pour, T. Rodemann, J. Hakanen, and K. Miettinen, “Surrogate assisted interactive
     multiobjective optimization in energy system design of buildings,” 
     Optimization and Engineering, 2021.'
    Args:
        reference_front (np.ndarray): The reference front that the current front is being compared to.
        Should be an one-dimensional array.
        individuals (np.ndarray): Current individuals generated by using surrogate models
        objectives (np.ndarray): Current objectives  generated by using surrogate models
        uncertainity (np.ndarray): Current Uncertainty values generated by using surrogate models
        problem : the problem class
    Returns:
        float: the new problem object that has an updated archive.
    """     
    
    nd = remove_duplicate(individuals, problem.archive.drop(
            problem.objective_names, axis=1).to_numpy()) #removing duplicate solutions
    if len(nd) == 0:
        return problem
    else:
        non_duplicate_dv = individuals[nd]
        non_duplicate_obj = objectives[nd]
        non_duplicate_unc = uncertainity[nd]
        
    # can try to use ExperimentalProblems archive, assuming it works properly for now.
    # TODO: make sure prob archive gets updated here with exact evals

   
    # just evaluate all for now
    problem.evaluate(non_duplicate_dv, use_surrogate=False)[0]

    # offline 
    # problem.evaluate(non_duplicate_dv, use_surrogate=True)[0]
        # Selecting solutions with lowest ASF values

    # online, update maxEI_index,solutions
    #problem.evaluate(non_duplicate_dv[maxEI_index], use_surrogate=False)[0]

    # update the model   
    problem.train(models=GaussianProcessRegressor,\
         model_parameters={'kernel': Matern(nu=1.5)}) 

    return problem

class RVEA(BaseDecompositionEA):
    """The python version reference vector guided evolutionary algorithm.
    Most of the relevant code is contained in the super class. This class just assigns
    the APD selection operator to BaseDecompositionEA.
    NOTE: The APD function had to be slightly modified to accomodate for the fact that
    this version of the algorithm is interactive, and does not have a set termination
    criteria. There is a time component in the APD penalty function formula of the type:
    (t/t_max)^alpha. As there is no set t_max, the formula has been changed. See below,
    the documentation for the argument: penalty_time_component
    See the details of RVEA in the following paper
    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
    Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
    Evolutionary Computation, 2016
    Parameters
    ----------
    problem : MOProblem
        The problem class object specifying the details of the problem.
    population_size : int, optional
        The desired population size, by default None, which sets up a default value
        of population size depending upon the dimensionaly of the problem.
    population_params : Dict, optional
        The parameters for the population class, by default None. See
        desdeo_emo.population.Population for more details.
    initial_population : Population, optional
        An initial population class, by default None. Use this if you want to set up
        a specific starting population, such as when the output of one EA is to be
        used as the input of another.
    alpha : float, optional
        The alpha parameter in the APD selection mechanism. Read paper for details.
    lattice_resolution : int, optional
        The number of divisions along individual axes in the objective space to be
        used while creating the reference vector lattice by the simplex lattice
        design. By default None
    a_priori : bool, optional
        A bool variable defining whether a priori preference is to be used or not.
        By default False
    interact : bool, optional
        A bool variable defining whether interactive preference is to be used or
        not. By default False
    n_iterations : int, optional
        The total number of iterations to be run, by default 10. This is not a hard
        limit and is only used for an internal counter.
    n_gen_per_iter : int, optional
        The total number of generations in an iteration to be run, by default 100.
        This is not a hard limit and is only used for an internal counter.
    total_function_evaluations :int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    penalty_time_component: Union[str, float], optional
        The APD formula had to be slightly changed.
        If penalty_time_component is a float between [0, 1], (t/t_max) is replaced by
        that constant for the entire algorithm.
        If penalty_time_component is "original", the original intent of the paper is
        followed and (t/t_max) is calculated as
        (current generation count/total number of generations).
        If penalty_time_component is "function_count", (t/t_max) is calculated as
        (current function evaluation count/total number of function evaluations)
        If penalty_time_component is "interactive", (t/t_max)  is calculated as
        (Current gen count within an iteration/Total gen count within an iteration).
        Hence, time penalty is always zero at the beginning of each iteration, and one
        at the end of each iteration.
        Note: If the penalty_time_component ever exceeds one, the value one is used as
        the penalty_time_component.
        If no value is provided, an appropriate default is selected.
        If `interact` is true, penalty_time_component is "interactive" by default.
        If `interact` is false, but `total_function_evaluations` is provided,
        penalty_time_component is "function_count" by default.
        If `interact` is false, but `total_function_evaluations` is not provided,
        penalty_time_component is "original" by default.
    """

    def __init__(
        self,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        alpha: float = 2,
        lattice_resolution: int = None,
        selection_type: str = None,
        #a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        time_penalty_component: Union[str, float] = None,
        keep_archive: bool = True
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            #a_priori=a_priori,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            keep_archive = keep_archive,
        )
        self.time_penalty_component = time_penalty_component
        self.objs_interation_end = None
        self.unc_interaction_end = None
        time_penalty_component_options = ["original", "function_count", "interactive"]
        if time_penalty_component is None:
            if interact is True:
                time_penalty_component = "interactive"
            elif total_function_evaluations > 0:
                time_penalty_component = "function_count"
            else:
                time_penalty_component = "original"
        if not (type(time_penalty_component) is float or str):
            msg = (
                f"type(time_penalty_component) should be float or str"
                f"Provided type: {type(time_penalty_component)}"
            )
            eaError(msg)
        if type(time_penalty_component) is float:
            if (time_penalty_component <= 0) or (time_penalty_component >= 1):
                msg = (
                    f"time_penalty_component should either be a float in the range"
                    f"[0, 1], or one of {time_penalty_component_options}.\n"
                    f"Provided value = {time_penalty_component}"
                )
                eaError(msg)
            time_penalty_function = self._time_penalty_constant
        if type(time_penalty_component) is str:
            if time_penalty_component == "original":
                time_penalty_function = self._time_penalty_original
            elif time_penalty_component == "function_count":
                time_penalty_function = self._time_penalty_function_count
            elif time_penalty_component == "interactive":
                time_penalty_function = self._time_penalty_interactive
            else:
                msg = (
                    f"time_penalty_component should either be a float in the range"
                    f"[0, 1], or one of {time_penalty_component_options}.\n"
                    f"Provided value = {time_penalty_component}"
                )
                eaError(msg)
        self.time_penalty_function = time_penalty_function
        self.alpha = alpha
        self.selection_type = selection_type
        selection_operator = APD_Select(
            pop=self.population,
            time_penalty_function=self.time_penalty_function,
            alpha=alpha,
            selection_type=selection_type,
        )
        self.selection_operator = selection_operator

    def _time_penalty_constant(self):
        """Returns the constant time penalty value.
        """
        return self.time_penalty_component

    def _time_penalty_original(self):
        """Calculates the appropriate time penalty value, by the original formula.
        """
        return self._current_gen_count / self.total_gen_count

    def _time_penalty_interactive(self):
        """Calculates the appropriate time penalty value.
        """
        return self._gen_count_in_curr_iteration / self.n_gen_per_iter

    def _time_penalty_function_count(self):
        """Calculates the appropriate time penalty value.
        """
        return self._function_evaluation_count / self.total_function_evaluations
    
    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.
        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.
        """
        if not isinstance(preference, (ReferencePointPreference, type(None))):
            msg = (
                f"Wrong object sent as preference. Expected type = "
                f"{type(ReferencePointPreference)} or None\n"
                f"Recieved type = {type(preference)}"
            )
            raise eaError(msg)
        if preference is not None:
            if preference.request_id != self._interaction_request_id:
                msg = (
                    f"Wrong preference object sent. Expected id = "
                    f"{self._interaction_request_id}.\n"
                    f"Recieved id = {preference.request_id}"
                )
                raise eaError(msg)
        #if preference is None and not self._ref_vectors_are_focused:
            #print("adapting")
            #self.reference_vectors.adapt(self.population.fitness)
        if preference is not None:
            ideal = self.population.ideal_fitness_val
            #fitness_vals = self.population.ob
            refpoint_actual = (
                preference.response.values * self.population.problem._max_multiplier
            )
            refpoint = refpoint_actual - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            """
            # evaluate alpha_k
            cos_theta_f_k = self.reference_vectors.find_cos_theta_f_k(refpoint_actual, self.population, self.objs_interation_end, self.unc_interaction_end)
            # adapt reference vectors
            self.reference_vectors.interactive_adapt_offline_adaptive(refpoint, cos_theta_f_k)
            """
            #self.reference_vectors.iteractive_adapt_1(refpoint)
            self.reference_vectors.add_edge_vectors()
        self.reference_vectors.neighbouring_angles()



class ProbRVEA(RVEA):
    def __init__(
        self,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        alpha: float = 2,
        lattice_resolution: int = None,
        #a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        time_penalty_component: Union[str, float] = None,
        keep_archive: bool = True
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            #a_priori=a_priori,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            keep_archive = keep_archive,
        )
        num_of_fit = problem.n_of_fitnesses
        num_of_obj = len(problem.objective_names)
        print("num of obj", num_of_obj)
        selection_operator = Prob_APD_select(self.population, num_of_fit, num_of_obj, self.time_penalty_function, alpha)
        # 9.4 60
        #MINMAXES
        #(-0.5418069650210939, 1.6057726473978304)
        #(-0.38166302048504885, 1.6335224950810368)
        #(-0.3829086159722088, 1.391068156873832)

        #selection_operator = Prob_APD_select_v3(self.population, num_of_fit, num_of_obj, self.time_penalty_function, alpha)

        # 9.2, 62
        #MINMAXES
        # : (-0.36064886128615825, 1.5298248022559235)
        #: (-0.36272337793604237, 1.5173115800959052)
        #: (-0.25439010545210294, 1.3538371598859698)
        

        self.selection_operator = selection_operator


    def iterate(self):
        super().iterate()
        #print(self.reference_vectors)
        #updated_problem = rvea_mm(
        #self.reference_vectors,
        #self.population.individuals,
        #self.population.objectives,
        #self.population.uncertainity,
        #self.population.problem)
        #self.number_of_update)
        #self.population.problem = updated_problem


    def _next_gen(self):
        """Run one generation of decomposition based EA. Intended to be used by
        next_iteration.
        """
        offspring = self.population.mate()  # (params=self.params)
        self.population.add(offspring, self.use_surrogates)
        selected = self._prob_select()
        self.population.keep(selected)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        if not self.use_surrogates:
            self._function_evaluation_count += offspring.shape[0]      


    def _prob_select(self) -> list:
        """Describe a selection mechanism. Return indices of selected
        individuals.

        Returns
        -------
        list
            List of indices of individuals to be selected.
        """
        return self.selection_operator.do(self.population, self.reference_vectors)

        
if __name__=="__main__":
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    prr = "dtlz4"

    def obj_function1(x):

        out = {
        "F": "",
        "G": "",
        }
        problem = get_problem(prr, 10)
        problem._evaluate(x, out)
        return out['F'][:,0]
    def obj_function2(x):

        out = {
        "F": "",
        "G": "",
        }
        problem = get_problem(prr, 10)
        problem._evaluate(x, out)
        return out['F'][:,1]
    def obj_function3(x):

        out = {
        "F": "",
        "G": "",
        }
        problem = get_problem(prr, 10)
        problem._evaluate(x, out)
        return out['F'][:,2]


    refpoint = np.asarray([0.2,0.5,0.9])
    n_obj = 3
    n_var = n_obj + 7
    var_names = ["x" + str(i + 1) for i in range(n_var)]
    obj_names = ["f" + str(i + 1) for i in range(n_obj)]
    unc_names = ["unc" + str(i + 1) for i in range(n_obj)]
# fundumentals of problem:

#creating the initial population
    x = np.random.random((109, n_var))
    initial_obj = {
        "F": "",
        "G": "",
        }
    get_problem(prr, 10)._evaluate(x, initial_obj)

    data = np.hstack((x, initial_obj['F']))
    datapd = pd.DataFrame(data=data, columns=var_names+obj_names)

    problem = ExperimentalProblem(data = datapd, objective_names=obj_names, variable_names=var_names,\
         uncertainity_names=unc_names, evaluators = [obj_function1, obj_function2, obj_function3])
    problem.train(models=SurrogateKriging)
    evolver = ProbRVEA(
                problem, interact=False, n_iterations=10, n_gen_per_iter = 100,\
                     lattice_resolution=10, use_surrogates=True,  population_size= 109, total_function_evaluations=150)#, number_of_update=u)
    #problem.train(models=GaussianProcessRegressor, model_parameters={'kernel': Matern(nu=1.5)})

    while evolver.continue_iteration():
        evolver.iterate()

    #evolver.end()
    print(evolver.total_function_evaluations)
    obj = evolver.population.objectives
    obj1minmax = (np.min(obj[:,0]), np.max(obj[:,0]))
    obj2minmax = (np.min(obj[:,1]), np.max(obj[:,1]))
    obj3minmax = (np.min(obj[:,2]), np.max(obj[:,2]))

    print("MINMAXES\n:", obj1minmax)
    print(obj2minmax)
    print(obj3minmax)

    hv_evolver = hypervolume_indicator(obj, np.array([1.5,1.5,1.5]))
    print("HV of evolver", hv_evolver, len(obj))

    #y_f = front_u[:,[0,1,2]]
    #y_f = y_f.copy(order='C')
    objectives = problem.archive.drop(problem.variable_names, axis=1).to_numpy()
    Best_solutions = objectives

    hv_archive = hypervolume_indicator(objectives, np.array([1.5,1.5,1.5]))
    print("HV of archive", hv_archive, len(objectives))
    # hvs to beat
    #HV of evolver 9.225883017160772 58
    #HV of archive 7.127860863245687 167
    #HV of pf front 7.402240124013435 66

    import plotly.graph_objs as go
    from pymoo.factory import get_problem, get_reference_directions, get_visualization
    #the next three lines are to get the true pareto front
    p = get_problem(prr, 10)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    pf = p.pareto_front(ref_dirs)

    hv_pf = hypervolume_indicator(pf, np.array([1.5,1.5,1.5]))
    print("HV of pf front", hv_pf, len(pf))


    x = Best_solutions[:,0]
    y = Best_solutions[:,1]
    z = Best_solutions[:,2]

    xo = obj[:,0]
    yo = obj[:,1]
    zo = obj[:,2]


    trace1 = go.Scatter3d(x=x, y=y, z=z, mode="markers",)
    trace2 = go.Scatter3d(x=xo, y=yo, z=zo, mode="markers",)
    #trace2 = go.Scatter3d(x=[refpoint[0]], y=[refpoint[1]], z=[refpoint[2]], mode="markers")
    trace3 = go.Mesh3d(x=pf[:,0], y=pf[:,1], z=pf[:,2])
    fig = go.Figure(data = [trace1, trace2, trace3])
    fig.show()
