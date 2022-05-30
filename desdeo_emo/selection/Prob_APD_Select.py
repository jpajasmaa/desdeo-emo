import numpy as np
from warnings import warn
from typing import List, Callable

from numpy.core.fromnumeric import size, std
from desdeo_emo.selection.SelectionBase import SelectionBase, InteractiveDecompositionSelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors
from typing import TYPE_CHECKING
from desdeo_emo.utilities.ProbabilityWrong import Probability_wrong
import os
import matplotlib.pyplot as plt
from matplotlib import rc

os.environ["OMP_NUM_THREADS"] = "1"


"""
def expected_impr(x0, x_sample, y, model):
    mu, sigma = model.predict(x0, return_std=True)
    sigma = sigma.reshape(-1, 1)
    max_val = np.max(y)
    tradeoff = 0.01

    return EI(mu, sigma, max_val, tradeoff)

def propose_location(X_sample, Y_sample, model, bounds):
    min_x = np.array([1.,1.]) # atleast return something
    
    def min_obj(X0):
        # Minimization objective is the negative acquisition function
        return -expected_impr(X0, X_sample, Y_sample, model)
    
    # Find the best optimum by starting from n_restart different random points.   
    
    pop_s = Y_sample.shape[0]
    bounds = np.array([[-2, -2], [2, 2]]) # variable bounds (lower, upper)

    pop = create_samples(2, pop_s, bounds)
    ga = real_GA(min_obj, pop, pop_s, pm, bounds, di, order, fmax, gen_max)
    for _ in range(10):
        ga.run() # run ga for 20 iterations
    
    best_fit = np.argmin(ga.fitness)
    min_x = ga.pop[best_fit]      
    return min_x

        pseudo

        exact and non exact sub pop members:
         apd* = exact apd min
         apd = pwrog samples (1000)

         u(x) = max(0, apd* - apd_i) for 1000

         u(x) has 1000 values either 0 or bigger.

         Take average of the ux.. is that just np.mean? 

         not sure if i need to optimize EI here.. he did mention monte carlo at somepoint

         in any case. the average of ux == expectation of APD.
         either:
            optimize: max expectedAPD
         no subpopmembers?:
            select individual with the average of ux value ? just like with original.


        i feel like i guess i should optimize, but how since I dont have the distributions? just need to use samples, which i do have but i dont know what.

        ei = 0
        fmin = np.amin(apd_exact)
        #for i in range(len(apd_exact)):
        for j in range(len(apd)):
            temp_ei = fmin - apd[j]
            if temp_ei > ei:
                ei = temp_ei
"""




from scipy.stats import norm
from scipy.special import ndtr
def EI(mean, std, max_val, tradeoff):
    imp = (mean - max_val - tradeoff)
    z = imp / std
    ei = imp * norm.cdf(z) + std * norm.pdf(z)
    ei[std == 0.0] = 0.0
    return ei


class Prob_APD_select(InteractiveDecompositionSelectionBase):
    """The selection operator for the RVEA algorithm. Read the following paper for more
        details.
        R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
        Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
        Evolutionary Computation, 2016
    Parameters
    ----------
    pop : Population
        The population instance
    time_penalty_function : Callable
        A function that returns the time component in the penalty function.
    alpha : float, optional
        The RVEA alpha parameter, by default 2
    """

    def __init__(
        self, pop: Population, number_of_vectors, number_of_objectives, time_penalty_function: Callable, alpha: float = 2#, archive
    ):
        self.time_penalty_function = time_penalty_function
        self.alpha = alpha
        self.n_of_objectives = pop.problem.n_of_objectives
        self.vectors = ReferenceVectors(
            number_of_vectors=number_of_vectors,
            number_of_objectives=number_of_objectives,
        )
        self.ei_apd = False
        #self.arch = archive

    def do(self, pop: Population, vectors: ReferenceVectors) -> List[int]:
        
        fitness = pop.fitness
        uncertainty = pop.uncertainity
        penalty_factor = self._partial_penalty_factor()
        
        refV = vectors.neighbouring_angles_current
        fmin = np.amin(fitness, axis=0)
        translated_fitness = fitness - fmin
        samps = 1000 # should this have as many as individuals?
        pwrong = Probability_wrong(mean_values=translated_fitness, stddev_values=uncertainty, n_samples=samps)
        pwrong.vect_sample_f()

        fitness_norm = np.linalg.norm(pwrong.f_samples, axis=1)
        fitness_norm = np.repeat(np.reshape(fitness_norm, (len(fitness), 1, pwrong.n_samples)), len(fitness[0, :]), axis=1)

        normalized_fitness = np.divide(pwrong.f_samples, fitness_norm)  # Checked, works.
        #print("norm fit", normalized_fitness.shape)


        # TODO:
        # now runs but only one individual
        # could be bc EI not implemented
        #but could be also bc we use x,y,z dim matrices in original code and im just dropping the z, bc it kinda does not exist
        # can try adding it with just len(obj_archive), amount of individuals

        gc = str(pop.gen_count-1)
        ind_archive = pop.individuals_archive[gc]
        obj_archive = pop.objectives_archive[gc]
        uc_archive = pop.uncertainty_archive[gc]
        #print(obj_archive['1'])
        #print(gc)
        #print(obj_archive[gc])
        # consider the exact function evaluations in the archive and assign them to subpopulations
        exact_f = obj_archive
        #print(exact_f.shape)
        exactmin = np.amin(exact_f)
        #print(exactmin)
        translated_exact = exact_f - exactmin
       # print(translated_exact.shape)
        # do i have to do the normalization? now just doing it
        translated_norm = np.linalg.norm(translated_exact, axis=1)
        translated_norm = np.repeat(np.reshape(translated_norm, (len(obj_archive), 1)), len(obj_archive[0, :]), axis=1)
        normalized_exact = np.divide(exact_f,translated_norm)  # Checked, works.
        #print(normalized_exact.shape)

        #  calculate the APD of these exact individuals, and f* (or let's say APD*_{exact, j} is the minimum APD in the jth subpopulation

        #Find cosine angles for all the samples
        cosine = np.tensordot(normalized_fitness, np.transpose(vectors.values), axes=([1], [0]))
        cosine = np.transpose(cosine,(0,2,1))
        #print(cosine.shape)

        if cosine[np.where(cosine > 1)].size:
            cosine[np.where(cosine > 1)] = 1
        if cosine[np.where(cosine < 0)].size:
            cosine[np.where(cosine < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta = np.arccos(cosine)
        # Reference vector asub_population_indexssignment
        #pwrong.compute_pdf(cosine)
        # Compute rank of cos theta (to be vectorized)
        rank_cosine = np.mean(cosine,axis=2)
        #print("rank cos", rank_cosine.shape)
        assigned_vectors = np.argmax(rank_cosine, axis=1)
        
        # add selected to here
        selection = np.array([], dtype=int)

        # assign in subpopuls
        cosine_e = np.dot(normalized_exact, np.transpose(vectors.values))
        #cosine = np.transpose(cosine,(0,2,1))
        #print(cosine_e.shape)

        if cosine_e[np.where(cosine_e > 1)].size:
            cosine_e[np.where(cosine_e > 1)] = 1
        if cosine_e[np.where(cosine_e < 0)].size:
            cosine_e[np.where(cosine_e < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta_e = np.arccos(cosine_e)
        # Reference vector asub_population_indexssignment
        #pwrong.compute_pdf(cosine)
        # Compute rank of cos theta (to be vectorized)
        #rank_cosine_e = np.mean(cosine_e, axis=0)
        rank_cosine_e = cosine_e
        #print(rank_cosine_e.shape)
        assigned_vectors_e = np.argmax(rank_cosine_e, axis=1)


        vector_selection = None

        for i in range(0, len(vectors.values)):
            sub_population_index = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors == i))
            )
            sub_population_fitness = pwrong.f_samples[sub_population_index]
            #print(sub_population_fitness.shape)

            exact_sub_population_index = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors_e == i))
            )
            #print(sub_population_index, exact_sub_population_index)
            exact_sub_population_fitness = exact_f[exact_sub_population_index]
            uc_ex = uc_archive[exact_sub_population_index]


            # ^ this very often empty

            #print(exact_sub_population_fitness.shape) #[[xx,yy,zz]], (1,3) shape
            #print(exact_sub_population_fitness)

            #input()
            minidx = None

            if len(sub_population_fitness > 0):
                print("there are subpop members")
                # APD Calculation
                angles = theta[sub_population_index, i]
                angles = np.divide(angles, refV[i])  # This is correct.
                #print("anglen",angles.shape)
                # You have done this calculation before. Check with fitness_norm
                # Remove this horrible line
                sub_pop_fitness_magnitude = np.sqrt(
                    np.sum(np.power(sub_population_fitness, 2), axis=1)
                )
                #print(sub_pop_fitness_magnitude.shape)
                sub_popfm = np.reshape(sub_pop_fitness_magnitude, (1, len(sub_pop_fitness_magnitude[:,0]), pwrong.n_samples))
                angles = np.reshape(angles,(1,len(angles),pwrong.n_samples))

                apd = np.multiply(
                    sub_popfm,
                    (1 + np.dot(penalty_factor, angles))
                )

                safe_pick = np.mean(apd, axis=2)
                safeidx = np.where(safe_pick[0] == np.nanmin(safe_pick[0]))
                minidx = safeidx
                selx = sub_population_index[minidx]
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                    vector_selection = np.asarray(i)
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))
                    vector_selection = np.hstack((vector_selection, i))

                    # if exact sub pop has members
                if len(exact_sub_population_fitness > 0):
                    print("there are exact subpop members")
                    # f* eg APD*_{exact,j} calculation
                    exact_angles = theta_e[exact_sub_population_index, i]
                    # uc_ex = theta_e[exact_sub_population_index, i] # this gets uc dims right but prob does not make any sense
                    #uc_ex = uc_ex[exact_sub_population_index] # this does not work
                    exact_angles = np.divide(exact_angles, refV[i])  # This is correct.
                    #print("exact ang",exact_angles.shape)
                    exact_sub_pop_fitness_magnitude = np.sqrt(
                        np.sum(np.power(exact_sub_population_fitness, 2), axis=1)
                    )
                    #print(exact_sub_pop_fitness_magnitude.shape)
                    e_sub_popfm = np.reshape(exact_sub_pop_fitness_magnitude, (1, len(exact_sub_pop_fitness_magnitude)))
                    apd_exact = np.multiply(e_sub_popfm, (1 + np.dot(penalty_factor, exact_angles)))
                    fmin = np.amax(apd_exact) # do we want max or min of real apd values

                    ux = np.zeros((sub_popfm.shape[1],1000))
                    for i in range(apd.shape[1]):
                        for j in range(apd.shape[2]):
                            ux[i][j] = np.max((0., fmin - apd[:, i, j]))

            
                    #b = [(0,np.max(ux))]
                    #print(b)
                    #def opt_ei(x,ux):
                    #    return -ux*x

                    #from scipy.optimize import differential_evolution
                    #res = differential_evolution(opt_ei, b, ux)
                    #print(res.x, res.fun)
                    #input()


                    ux = ux.reshape((1,sub_popfm.shape[1],1000))
                    impromean = np.mean(ux, axis=2) # optimize instead of this?
                    impromax = np.max(ux, axis=2) # optimize instead of this?
                    print(impromean, impromax)
                    rank_apd = impromax
                    #rank_apd = impromean
                    #print("eapd shape", rank_apd.shape)

                    minidx = np.where(rank_apd[0] == np.nanmax(rank_apd[0]))[0]
                    #exact_minidx = np.where(rank_apd[0,:] == np.nanmin(rank_apd[0,:]))[0][0]
                    print("min IDXZ",minidx)
                    # minidx = np.where(rank_apd[0] == np.nanmin(rank_apd[0]))

                    # now inside exact
                    #if np.isnan(apd).all():
                    #    continue
                    selx = sub_population_index[minidx]
                    if selection.shape[0] == 0:
                        selection = np.hstack((selection, np.transpose(selx[0])))
                        vector_selection = np.asarray(i)
                    else:
                        selection = np.vstack((selection, np.transpose(selx[0])))
                        vector_selection = np.hstack((vector_selection, i))

        if selection.shape[0] == 0:
            print("GOT here")
            rand_select = np.random.randint(len(fitness), size=1)
            selection = rand_select

        if selection.shape[0] == 1:
            print("Only one individual!!")
            rand_select = np.random.randint(len(fitness), size=1)
            selection = np.vstack((selection, np.transpose(rand_select[0])))

        #print("SELECTION:\n", selection)
        print(selection.shape)
        #res = selection.squeeze()
        #print(res)

        return selection.squeeze()




    def _partial_penalty_factor(self) -> float:
        """Calculate and return the partial penalty factor for APD calculation.
            This calculation does not include the angle related terms, hence the name.
            If the calculated penalty is outside [0, 1], it will round it up/down to 0/1
        Returns
        -------
        float
            The partial penalty value
        """
        if self.time_penalty_function() < 0:
            px = 0
        elif self.time_penalty_function() > 1:
            px = 1
        else:
            px= self.time_penalty_function()
        penalty = ((px) ** self.alpha) * self.n_of_objectives

        return penalty

    def adapt_RVs(self, fitness: np.ndarray) -> None:
        self.vectors.adapt(fitness)
        self.vectors.neighbouring_angles()
        #vectors = self.vectors


class Prob_APD_select_v3(SelectionBase):
    """The selection operator for the RVEA algorithm. Read the following paper for more
        details.
        R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
        Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
        Evolutionary Computation, 2016
    Parameters
    ----------
    pop : Population
        The population instance
    time_penalty_function : Callable
        A function that returns the time component in the penalty function.
    alpha : float, optional
        The RVEA alpha parameter, by default 2
    """

    def __init__(
        self, pop: Population, number_of_vectors, number_of_objectives, time_penalty_function: Callable, alpha: float = 2
    ):
        self.time_penalty_function = time_penalty_function
        self.alpha = alpha
        self.n_of_objectives = pop.problem.n_of_objectives
        self.vectors = ReferenceVectors(
            number_of_vectors=number_of_vectors,
            number_of_objectives=number_of_objectives,
        )

    def do(self, pop: Population, vectors: ReferenceVectors) -> List[int]:


        #def Prob_APD_select_v3(
        #    fitness: list,
        #    uncertainty: list,
        #    vectors: "ReferenceVectors",
        #    penalty_factor: float,
        #    ideal: list = None,
        #):
        fitness = pop.fitness
        uncertainty = pop.uncertainity
        penalty_factor = self._partial_penalty_factor()
        refV = vectors.neighbouring_angles_current
        fmin = np.amin(fitness, axis=0)
        translated_fitness = fitness - fmin
        pwrong = Probability_wrong(mean_values=translated_fitness, stddev_values=uncertainty, n_samples=1000)
        pwrong.vect_sample_f()

        fitness_norm = np.linalg.norm(pwrong.f_samples, axis=1)
        fitness_norm = np.repeat(np.reshape(fitness_norm, (len(fitness), 1, pwrong.n_samples)), len(fitness[0, :]), axis=1)

        normalized_fitness = np.divide(pwrong.f_samples, fitness_norm)  # Checked, works.


        #Find cosine angles for all the samples
        cosine = np.tensordot(normalized_fitness, np.transpose(vectors.values), axes=([1], [0]))
        cosine = np.transpose(cosine,(0,2,1))

        if cosine[np.where(cosine > 1)].size:
            cosine[np.where(cosine > 1)] = 1
        if cosine[np.where(cosine < 0)].size:
            cosine[np.where(cosine < 0)] = 0
        # Calculation of angles between reference vectors and solutions
        theta = np.arccos(cosine)
        # Reference vector asub_population_indexssignment
        #pwrong.compute_pdf(cosine)
        # Compute rank of cos theta (to be vectorized)
        rank_cosine = np.mean(cosine,axis=2)
        assigned_vectors = np.argmax(rank_cosine, axis=1)
        selection = np.array([], dtype=int)

        vector_selection = None

        for i in range(0, len(vectors.values)):
            sub_population_index = np.atleast_1d(
                np.squeeze(np.where(assigned_vectors == i))
            )
            sub_population_fitness = pwrong.f_samples[sub_population_index]

            if len(sub_population_fitness > 0):
                # APD Calculation
                angles = theta[sub_population_index, i]
                angles = np.divide(angles, refV[i])  # This is correct.
                # You have done this calculation before. Check with fitness_norm
                # Remove this horrible line
                sub_pop_fitness_magnitude = np.sqrt(
                    np.sum(np.power(sub_population_fitness, 2), axis=1)
                )
                sub_popfm = np.reshape(sub_pop_fitness_magnitude, (1, len(sub_pop_fitness_magnitude[:,0]), pwrong.n_samples))
                angles = np.reshape(angles,(1,len(angles),pwrong.n_samples))


                #### Overall Mean/Median of apd
                apd = np.multiply(
                    sub_popfm,
                    (1 + np.dot(penalty_factor, angles))
                )
                rank_apd = np.mean(apd, axis=2)
                print(rank_apd)
                minidx = np.where(rank_apd[0] == np.nanmin(rank_apd[0]))

                if np.isnan(apd).all():
                    continue
                selx = sub_population_index[minidx]
                if selection.shape[0] == 0:
                    selection = np.hstack((selection, np.transpose(selx[0])))
                    vector_selection = np.asarray(i)
                else:
                    selection = np.vstack((selection, np.transpose(selx[0])))
                    vector_selection = np.hstack((vector_selection, i))

        if selection.shape[0] == 1:
            print("Only one individual!!")
            rand_select = np.random.randint(len(fitness), size=1)
            selection = np.vstack((selection, np.transpose(rand_select[0])))
        print("Selection:",selection.shape)
        return selection.squeeze()

    def _partial_penalty_factor(self) -> float:
        """Calculate and return the partial penalty factor for APD calculation.
            This calculation does not include the angle related terms, hence the name.
            If the calculated penalty is outside [0, 1], it will round it up/down to 0/1
        Returns
        -------
        float
            The partial penalty value
        """
        penalty = ((self.time_penalty_function()) ** self.alpha) * self.n_of_objectives
        if penalty < 0:
            penalty = 0
        if penalty > 1:
            penalty = 1
        return penalty


    def adapt_RVs(self, fitness: np.ndarray) -> None:
        self.vectors.adapt(fitness)
        self.vectors.neighbouring_angles()
        #vectors = self.vectors



