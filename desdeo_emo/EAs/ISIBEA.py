from typing import Dict, Callable
import numpy as np
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.TournamentSelection import TournamentSelection
from desdeo_tools.scalarization import SimpleASF
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA
from desdeo_tools.utilities.quality_indicator import preference_indicator 
from desdeo_problem import MOProblem

from desdeo_tools.utilities import fast_non_dominated_sort, fast_non_dominated_sort_indices


import hvwfg as hv


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

# dis needs to be able to calculate the loss too
def hypervolume(obj, ref):
    return hv.wfg(obj, ref)


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




"""
    I-SIBEA can do a-priori, interactive and post-eriori. I guess lets start with post-eriori and worry rest laterself.
    - a priori means we interact once with the problem, interactive is that we interact x times, x is what Dm decides
    - post-eriori is no interactions..

    NP =  pop size

    DA = max number of sols to be shown to DM (default 5)
    AA, RA == preferred and non-preferred sols after interaction (a-priori and interactive)
    H = max number of interactions (0 for post-eriori)


    Create pop, 
    do crossover and mutation to offspring Q
    COmbine P : P + Q, select to NP with enviromental selection
    do nondom sorting, add to P1 until P1 size is NP. Set P : P1 for next gen.
        - otherwise the set with worst rank in P1 is denoted P'. 
            Use hv selection for each z in P', the loss in the 
            hv d(z) = I(P') - I(P´ \ z) is determined. I is weighted hv indicator. Solution with the smallest loss
            is removed until the size of the pop does no longer exceed NP, then set P: P1.
    
    After fixed number of gens the interactions
    Step 5 show DA solutions to dm. ( so eg. clustering)
    Then final solution we optimize ASF.

    
    OK:
    TODO: find out how to do
     issue with baseEAs _next_gen with enviromental selection and using the worst index as means to removing from pop.
    -ISIBEA has different ways of doing this but BaseIndicatorEA does not have the way of doing this. 

    Also not 100 sure but does fitness assignment come different place in ISIBEA ?

"""


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
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        #indicator: Callable = weighted_hv_indicator,
        indicator: Callable = hypervolume,
        da: int = 5,
        population_params: Dict = None,
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

        self.indicator = indicator
        selection_operator = TournamentSelection(self.population, 2)
        self.selection_operator = selection_operator


    # override baseindiea
    def _next_gen(self):
        # step 2. Mating. 
        offspring = self.population.mate()
        # add works great
        self.population.add(offspring)
        #print(self.population.pop_size)
        #print(self.population.individuals.shape[0])
        
        # step 3. enviromental selection
        fronts_indices = fast_non_dominated_sort_indices(self.population.objectives)
        # not sure which i ll use
        #fronts = fast_non_dominated_sort(self.population.objectives)
        
        # add pop to P1 until |P1| >= pop.size
        # is cool but matlab code seems not to care about this
        P1 = self.add_to_P1(fronts_indices)
        #print(P1)
        #if len(P1) == NP

        self.fitnesses = self.population.objectives
        loss = np.zeros(20)
        # compute Hv indicator values 
        ref = np.array([5.0,5.0,5.0])
        fitness = np.ma.array(self.fitnesses, mask=False)
        # what i was doing with the masks ?
        # this mess actually might be ok. does step 3,c.
        for i in range(20):
            fitness.mask[i] = True
            lossval = hypervolume(self.fitnesses, ref) - hypervolume(fitness, ref)
            #if lossval == # could do check for too small values and make them to be say 0.0001 etc. do it if overflow/underflow errors
            loss[i] = lossval
            fitness.mask[i] = False

        print(loss)
        #NoN = 


        input()

        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]

    # this should work good enough for the step 3a.
    def add_to_P1(self, fronts):
        NP = 10
        P1 = []
        i=0 
        P_idx = []
        while len(P1) < NP and i <= len(fronts):    
            vals = fronts[i] # get Fi
            sols = vals[0][0:]
            P_idx.append(sols) # set front idxes to P1idx
            
            for j in sols:
                P1.append(self.population.objectives[j])
            i += 1
        
        P1 = np.stack(P1) # to return as 1 2d arr
        return P1


    
    # override baseindiea
    def _select(self):
        pass


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



if __name__=="__main__":
    print("ISIBEA")
    from desdeo_problem import MOProblem
    import matplotlib.pyplot as plt
    from desdeo_problem.testproblems.TestProblems import test_problem_builder

    # start the problem
    problem_name = "DTLZ4"
    problem = test_problem_builder(problem_name, n_of_variables=5, n_of_objectives=3)

    weights = [1,1,1]

    isibea = ISIBEA(problem, population_size=10, n_iterations=1, n_gen_per_iter=50,total_function_evaluations=1000, da=5 )
    while isibea.continue_evolution():
        isibea.iterate()
    individuals, objective_values = isibea.end()
    print("IBEA ideal",isibea.population.problem.ideal)
    plt.scatter(x=objective_values[:,0], y=objective_values[:,1], label="IBEA Front")
    plt.title(f"IBEA approximation")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    #plt.show()
    # need to get the population
    #ini_pop = ib.population # so pbea doesn't get only the non dom pop members, and popsize stays the same
