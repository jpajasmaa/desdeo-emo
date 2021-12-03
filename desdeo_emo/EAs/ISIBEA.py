from typing import Dict, Callable
import numpy as np
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.TournamentSelection import TournamentSelection
from desdeo_tools.scalarization import SimpleASF
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA
from desdeo_tools.utilities.quality_indicator import preference_indicator 
from desdeo_problem import MOProblem


"""
    SIBEA algo notes
    1. P pop, size of NP gen randomly
    2. crossover and mutation to create offspring pop Q size NP
    3. P : P + Q, combine parent and offspring pop, then do enviromental selection (non dom sorting) to reduce P to size on NP.
    4. DM interacts with the algo
    5. fixed number of nondom DA subset of A is identified (use k-means clustering)
    6. show DM the DA solutions of A 
    7. Terminate if dm wants to quit, show itself, use ASF.
    8. ask the DM to classify DA into AA and/or RA and derive the sets Do, In, Pr to get updated weighted hypervolume. go step 2.



    weight distribution function

    w(z) = 0 for a z in do
    

    weighted hv indicator: is calculated as the integral over the product of the weight distribution function and attainment_function

    Iw_hv = integral_0^1  w(z) * att(z) dz


    TODO:
    weighted hv seems bit complicated..
    imo two ways to simplify atleast to get started
    - only worry about 2d problems
    - ignore weighting, do first with hv from wfg. So its more like og sibea but only with the interaction parts by not liking some solutions etc.
    
    After either of those work can worry about the weighted hv.

"""

def weakly_dominates(v1, v2):
    pass 

# not quite sure if needed and where but
def attainment_function(z, set):
    # A >= (weakly dominates z)
    if weakly_dominates(set, z):
        return 1
    return 0



def weighted_hv_indicator():
    pass





class ISIBEA(BaseIndicatorEA):
    """Python Implementation of ISIBEA. 

    Most of the relevant code is contained in the super class. This class just assigns
    the EnviromentalSelection operator to BaseIndicatorEA.

    Parameters
    ----------
    problem: MOProblem
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
    total_function_evaluations : int, optional
        Set an upper limit to the total number of function evaluations. When set to
        zero, this argument is ignored and other termination criteria are used.
    use_surrogates: bool, optional
    	A bool variable defining whether surrogate problems are to be used or
        not. By default False
    kappa : float, optional
        Fitness scaling value for indicators. By default 0.05.
    indicator : Callable, optional
        Quality indicator to use in indicatorEAs. 
    reference_point : np.ndarray
        The reference point that guides the PBEAs search.
    delta : float, optional
        Spesifity for the preference based quality indicator. By default 0.01.

    """
    def __init__(self,
        problem: MOProblem,
        population_size: int,
        initial_population: Population,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        kappa: float = 0.05,
        indicator: Callable = weighted_hv_indicator,
        population_params: Dict = None,
        reference_point = None,
        delta: float = 0.1, 
        ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            a_priori=a_priori,
            interact=interact,
            n_iterations=n_iterations,
            initial_population=initial_population,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            use_surrogates=use_surrogates,
        )

        self.kappa = kappa
        self.delta = delta
        self.indicator = indicator
        self.reference_point = reference_point
        selection_operator = TournamentSelection(self.population, 2)
        self.selection_operator = selection_operator


    def _fitness_assignment(self, fitnesses):
        """
            Performs the fitness assignment of the individuals.
        """
        pass



    def _environmental_selection(self, fitnesses):
        """
            Selects the worst member of population, then updates the population members fitness values compared to the worst individual.
            Worst individual is removed from the population.
            
        """
        pass 

