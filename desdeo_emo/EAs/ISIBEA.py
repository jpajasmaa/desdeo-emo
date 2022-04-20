from typing import Dict, Callable, Tuple
import numpy as np
import pandas as pd
from desdeo_emo.population.Population import Population
from desdeo_tools.scalarization import SimpleASF
from desdeo_emo.EAs.BaseIndicatorEA import BaseIndicatorEA
from desdeo_problem import MOProblem

from desdeo_tools.utilities import fast_non_dominated_sort, fast_non_dominated_sort_indices
from desdeo_tools.interaction import (
   SimplePlotRequest,
   ReferencePointPreference,
    validate_ref_point_with_ideal_and_nadir,
)

import hvwfg as hv

def show_clustered_solutions(objective_values, da):
    codebook, distortion = kmeans(objective_values, da)
    plt.scatter(codebook[:,0], codebook[:,1], c='r')
    plt.title(f"I-SIBEA DA clusters")
    plt.xlabel("F1")
    plt.ylabel("F2")
    #plt.legend()
    plt.show()
    return codebook


def plot_final(objective_values, chosen):
    if objective_values.shape[1] == 2:
        plot_final2d(objective_values, chosen)
    if objective_values.shape[1] == 3:
        plot_final3d(objective_values, chosen)


def plot_final2d(objective_values, chosen):
    plt.scatter(x=objective_values[:,0], y=objective_values[:,1], label="I-SIBEA Front")
    plt.scatter(x=chosen[0], y=chosen[1], c='r',s=100, label="Chosen best solution")
    plt.title(f"I-SIBEA approximation")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    plt.show()

def plot_final3d(objective_values, chosen):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(30,45)
    ax.scatter(objective_values[:,0],objective_values[:,1],objective_values[:,2], label="I-SIBEA Front")
    ax.scatter(chosen[0], chosen[1], chosen[2], s=100, c='r', label="Chosen best solution")
    plt.title(f"I-SIBEA approximation")
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_zlabel('F3')
    ax.legend()
    plt.show()
    

def tcheby(ref, obj, ideal):
    feval = np.abs(obj - ideal) * ref
    return np.max(feval)

# dis needs to be able to calculate the loss too
def hypervolume(obj, ref):
    return hv.wfg(obj, ref)


class ISIBEA(BaseIndicatorEA):

    def __init__(self,
        problem: MOProblem,
        population_size: int,
        initial_population: Population = None,
        a_priori: bool = False,
        interact: bool = False, # false means posteori I-SIBEA
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        use_surrogates: bool = False,
        indicator: Callable = hypervolume,
        da: int = 5,
        weights = [],
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
        self.da = da
        self.weights = weights
        #self.selection_operator = selection_operator
        self.fitnesses = self.population.objectives


    # override baseindiea
    def _next_gen(self):
        # step 2. Mating. 
        offspring = self.population.mate()
        # add works great
        self.population.add(offspring)
        
        # front indices has the non dominated solution indices
        fronts_indices = fast_non_dominated_sort_indices(self.population.objectives)
        
        non_dom_indices = []
        for i in fronts_indices:
            i = i[0] # take first of the tuple which has the indices
            for j in range(len(i)):
                non_dom_indices.append(i[j])

        NP = self.population.pop_size
        P1 = self.add_to_P1(fronts_indices)
        #print("P1", len(P1))
        
        #loss = None
        assert(len(P1) >= NP)

        if len(P1) == NP:
            self.population.keep(P1[:self.population.pop_size])
            print("P1 == NP")
        else:
            
            # TODO: make this use actual DM preferences
            
            # now only posteori way with tcheby being the DM.
            if self._current_gen_count in [50,150,300,350]:
                print("Interaction step with ADM")
                # TODO: this only missing. Use tseby when posteori and get from user when interactive..
                A_indices = np.array([i for i in range(self.fitnesses.shape[0])])
                xlist = []
                for i in A_indices:
                    xlist.append(tcheby(self.fitnesses[i], self.weights, self.population.ideal_objective_vector))
                
                AA = np.argmin(xlist)
                RA_all = A_indices
                RA = np.delete(RA_all, AA, 0)

                #fitnesses = self.fitnesses[:len(P1)]
                fitnesses = self.fitnesses
                loss = self._environmental_selection(fitnesses, AA, RA, IA)
            
            else:
            # step 3. enviromental selection
                #fitnesses = self.fitnesses[:len(P1)]
                fitnesses = self.fitnesses
                loss = self._environmental_selection(fitnesses)
             
            # TODO: make sure P1 has correct indis and get rid of computing all losses always
            # remove until pop size is the same as starting
            # 3d remove |P1| - NP solutions from P' with the smallest loss d(z)
            # include the remaining in P' to P1, set P = P1                
            diff = len(P1) - NP
            worst_idx = np.argsort(loss)[-diff:]
            P1 = np.delete(P1, worst_idx, 0)    
            # set P = P1         
            self.population.keep(P1)
            #self.fitnesses = self.population.objectives

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


    def _environmental_selection(self, fitnesses, AA=None, RA=None, IA=None):
        """
            
        """
        # atleast AA needs to exist i think.. atleast lest just consider it for starters
        
        #self.fitnesses = fitnesses
        #print(self.fitnesses[:5]) # smh wrong with the fitnesses
        loss = np.zeros(fitnesses.shape[0])
        #ref = np.max(self.population.objectives, axis=0)
        ref = np.max(fitnesses, axis=0)
        P_not_z = np.ma.array(fitnesses, mask=False) # Masks are to mask the z out.
        
        if AA is not None:
            # TODO: does not work, some reason second objective is very big.. this makes hv to be 0. somehow we break the self.fitnesses[1]..
            #print(self.fitnesses[:5])
            do_idx = np.array(RA)
            pr_idx = np.array(AA)
            mask_do = np.zeros(fitnesses.shape, bool)
            mask_pr = np.zeros(fitnesses.shape, bool)
            mask_do[do_idx] = True
            mask_pr[pr_idx] = True
            #print(mask_do)
            do = np.ma.masked_array(fitnesses, mask_do)
            pr = np.ma.masked_array(fitnesses, mask_pr)
            #print(fitnesses[:,1])
    
            hv_Do = hypervolume(do, ref)
            hv_Pr = hypervolume(pr, ref)
            
            divide = hv_Do/(hv_Pr+0.000001) # to prevent divide by 0
            #print(divide) # some possible issue here since it seems to be 0 very often          
            
            for i in range(fitnesses.shape[0]):
                P_not_z.mask[i] = True
                # d(z) = I(P') - I(P' \ z) . Now P' = P
                lossval = hypervolume(self.fitnesses, ref) - hypervolume(P_not_z, ref)
                w_z = 1 # default IA.
                if np.any(AA) == i: # if in preferred
                    w_z = 1 + divide # 1 + hv(DO)/hv(Pr)
                elif np.any(RA) == i: # if in non preferred
                    w_z = 0

                loss[i] = w_z * lossval
                P_not_z.mask[i] = False
        
        else:
            for i in range(fitnesses.shape[0]):
                P_not_z.mask[i] = True
                # d(z) = I(P') - I(P' \ z) . Now P' = P
                lossval = hypervolume(self.fitnesses, ref) - hypervolume(P_not_z, ref)
                loss[i] = lossval
                P_not_z.mask[i] = False
        
        # update self fitnesses
        self.fitnesses = fitnesses
        
        return loss


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
    
    
    def requests(self) -> Tuple:
        return (self.request_preferences(), self.request_plot())
    
    def request_preferences(self):
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        message = ("Please provide preferences as a preferred points (AA) or non-preferred points (RA). Atleast one point needs to be preferred or not preferred.\n\n"
            f"The preferred points will focus the search towards the preferred regions and non-preferred points away from the non-preferred regions. "
            )

        def validator(dimensions_data: pd.DataFrame, reference_point: pd.DataFrame):
            validate_ref_point_dimensions(dimensions_data, reference_point)
            validate_ref_point_data_type(reference_point)
            validate_ref_point_with_ideal(dimensions_data, reference_point)
            return
                   
        interaction_priority = "required"
        self._interaction_request_id = np.random.randint(0, 1e9)

        return ReferencePointPreference(
                dimensions_data=dimensions_data,
                message=message,
                interaction_priority=interaction_priority,
                preference_validator=validate_ref_point_with_ideal_and_nadir,
                request_id=self._interaction_request_id,
            
        )
        
        
    def request_plot(self) -> SimplePlotRequest:
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self.population.problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self.population.problem._max_multiplier
        dimensions_data.loc["ideal"] = self.population.ideal_objective_vector
        dimensions_data.loc["nadir"] = self.population.nadir_objective_vector
        data = pd.DataFrame(
            self.population.objectives, columns=self.population.problem.objective_names
        )
        return SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message="Objective Values"
        )


if __name__=="__main__":
    from desdeo_problem import MOProblem
    from desdeo_problem.testproblems.TestProblems import test_problem_builder
    from desdeo_tools.scalarization import SimpleASF # lets just use desdeos SimpleASF as the achievement scalarizing function

    problem_name = "ZDT4"
    problem = test_problem_builder(problem_name, n_of_variables=10, n_of_objectives=2)

    print("Solving problem: ", problem_name)
    weights = np.array([0.5,0.5])

    # run ibea
    # step 0. Let's start with rough approx
    isibea = ISIBEA(problem, population_size=100, n_iterations=10, n_gen_per_iter=100,total_function_evaluations=40000, da=5, interact=False, weights=weights)
    while isibea.continue_evolution():
        isibea.iterate()
        
        # DM, interactive version..
        if isibea._current_gen_count in [100,200,300,400] and isibea.interact == True:
            individuals, objective_values = isibea.end()
            print(isibea._gen_count_in_curr_iteration)
            print(isibea._current_gen_count) # this is the generations
            pref, plot = isibea.requests()
            print(pref.content['message'])
            
            # show DA solutions to DM
            #whitened = whiten(objective_values)
            da_sols = show_clustered_solutions(objective_values, isibea.da)
            print(da_sols)
            # TODO: handle selection
            
            #break
            
    individuals, objective_values = isibea.end()
    # if number of interactions == 0: do more complex thing otherwise
    we = 1/(np.max(objective_values, axis=0) - np.min(objective_values, axis=0)) # weight vector for asf is wi = 1 / zi_max - zi_min

    # posteriori maybe ok here
    if isibea.interact == False: # if posteori
        asf = SimpleASF(we) # weight vector is 0.5, 0.5
        asf_values = asf(objective_values, reference_point=weights)
        min_asf_idx = np.argmin(asf_values)
        min_asf_value = np.min(asf_values)
        #print(asf_values)
        #print(min_asf_value)
        print(f"Final solution after ASF: {objective_values[min_asf_idx]}")
        print(f"Mean of asf values: {np.mean(asf_values)}")

    # interactive seems to work ok too i guess
    else:       
        # show last da solutions
        da_sols = show_clustered_solutions(objective_values, isibea.da)
        print(da_sols)
        selected = da_sols[0] # we want the one in first index always
        asf = SimpleASF(weights) # weight vector is 0.5, 0.5
        asf_values = asf(selected, reference_point=weights)
        min_asf_idx = np.argmin(asf_values)
        min_asf_value = np.min(asf_values)
        #print(asf_values)
        #print(min_asf_value)
        print(f"Chosen solution after ASF: {objective_values[min_asf_idx]}")
        print(f"Mean of asf values: {np.mean(asf_values)}")

    chosen = objective_values[min_asf_idx]
    print(isibea._gen_count_in_curr_iteration)
    print(isibea._current_gen_count) # this is the generations
    print(isibea.total_function_evaluations) # total function evaluations
    plot_final(objective_values, chosen)
