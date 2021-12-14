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
        self.fitnesses = self.population.objectives


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
        #print("fit shape",self.fitnesses.shape)
        
        NP = self.population.pop_size
        #print("NP", NP)

        # TODO: what to do if P1 is smaller than NP, where to find pop members
        # TODO: control when P1 is smaller or greater than NP so neither popsize grows or shrinks during iterations.
        # add pop to P1 until |P1| >= pop.size
        # is cool but matlab code seems not to care about this
        P1 = self.add_to_P1(fronts_indices)
        #print("P1 len before",len(P1))

        assert(len(P1) >= NP)

        if len(P1) == NP:
            self.population.keep(P1[:self.population.pop_size])

        else:
            self.fitnesses = self.fitnesses[:len(P1)]
            loss = np.zeros(self.fitnesses.shape[0])
            # compute Hv indicator values 

            # nadir infinite is not good
            #ref = self.population.nadir_objective_vector # ref is like nadir at that point
            #ref = np.array([100.,100.,100.]) # ref is like nadir at that point
            ref = np.max(self.population.objectives, axis=0)
            P_not_z = np.ma.array(self.fitnesses, mask=False)
            # Masks are to mask the z out.
            # this mess actually might be ok. does step 3,c.
            for i in range(self.fitnesses.shape[0]):
                P_not_z.mask[i] = True
                # d(z) = I(P') - I(P' \ z) . Now P' = P
                lossval = hypervolume(self.fitnesses, ref) - hypervolume(P_not_z, ref)
                #if lossval == # could do check for too small values and make them to be say 0.0001 etc. do it if overflow/underflow errors
                loss[i] = lossval
                P_not_z.mask[i] = False

            #print(loss)
            
            # remove until pop size is the same as starting
            # 3d remove |P1| - NP solutions from P' with the smallest loss d(z)
            # include the remaining in P' to P1, set P = P1
            diff = len(P1) - NP

            #print("diff", diff)
            worst_idx = np.argsort(loss)[-diff:]
            #print(worst_idx)
                # same basic problem. need to return list with x worst values then pop them off.
            P1 = np.delete(P1, worst_idx, 0)    
                # worst_idx need to remove list that will tell which x number of idx to remove so pop size is NP again.
                # worst_index = np.argmin(loss, axis=0)
                # self.population.delete(worst_index)
                #self.fitnesses = np.delete(self.fitnesses, worst_index, 0)

            # set P = P1         
            self.population.keep(P1)
            #print("pop obj shape after",self.population.objectives.shape)
            #print("pop individuals shape after",self.population.individuals.shape)
            #print("POP sizes,", self.population.pop_size, self.population.individuals.shape[0])
            #worst_index = np.argmin(self.fitnesses, axis=0)[0] # gets the index worst member of population
            #self.fitnesses = self._environmental_selection(self.fitnesses, worst_index)
            # remove the worst member from population and from fitnesses
            #self.population.delete(worst_index)
            #self.fitnesses = np.delete(self.fitnesses, worst_index, 0)


        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        self._function_evaluation_count += offspring.shape[0]

    # this should work good enough for the step 3a.
    # we actually want to handle indixes here i think..
    def add_to_P1(self, fronts):
        NP = self.population.pop_size 
        i=0 
        P_idx = []
        while len(P_idx) < NP and i <= len(fronts):    
            vals = fronts[i] # get Fi
            sols = vals[0][0:]
            
            for j in (sols):
                P_idx.append(j)
            i += 1
       
        P_idx = np.asarray(P_idx)
        return P_idx


    
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


    # this might need override?
    def end(self):
        """Conducts non-dominated sorting at the end of the evolution process
        Returns:
            tuple: The first element is a 2-D array of the decision vectors of the non-dominated solutions.
                The second element is a 2-D array of the corresponding objective values.
        """
        non_dom = self.population.non_dominated_objectives()
        return (
            self.population.individuals[non_dom, :],
            self.population.objectives[non_dom, :],
        )


if __name__=="__main__":
    print("ISIBEA")
    from desdeo_problem import MOProblem
    import matplotlib.pyplot as plt
    from desdeo_problem.testproblems.TestProblems import test_problem_builder

    problem_name = "ZDT1"
    problem = test_problem_builder(problem_name)   # start the problem
    #problem_name = "DTLZ4"
    #problem = test_problem_builder(problem_name, n_of_variables=5, n_of_objectives=3)

    weights = [1,1,1]

    isibea = ISIBEA(problem, population_size=32, n_iterations=10, n_gen_per_iter=100,total_function_evaluations=5000, da=5 )
    print(isibea.population.objectives)
    while isibea.continue_evolution():
        isibea.iterate()
    individuals, objective_values = isibea.end()

    # TODO: rest of ISIBEA 
    # NOTE: baseEA end removes dominated pop members, thats why usually the amount individuals in the end is less than starting pop size.
   
    print("individuals shape", individuals.shape)
    print(objective_values.shape)
    #print("IBEA ideal",isibea.population.problem.ideal)
    plt.scatter(x=objective_values[:,0], y=objective_values[:,1], label="IBEA Front")
    plt.title(f"IBEA approximation")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    plt.show()
    # need to get the population
    #ini_pop = ib.population # so pbea doesn't get only the non dom pop members, and popsize stays the same
