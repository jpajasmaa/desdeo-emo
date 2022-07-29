from typing import Dict, Union

from desdeo_emo.EAs.BaseEA import BaseDecompositionEA, eaError
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.APD_Select_constraints import APD_Select
from desdeo_emo.selection.Prob_APD_Select import Prob_APD_select #, Prob_APD_select_v3  # superfast y considering mean APD
from desdeo_problem import MOProblem

import numpy as np




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

        #selection_operator = Prob_APD_select_v3(self.population, num_of_fit, num_of_obj, self.time_penalty_function, alpha)

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
        return self.selection_operator.do(self.population)
