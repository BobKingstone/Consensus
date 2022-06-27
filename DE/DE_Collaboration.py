import numpy as np
from DE.Individual_Collab import Individual_Consensus


class DE_Collaboration():

    def __init__(self,
                scaling_rate = 0.5,
                crossover_rate = 0.5,
                population_size = 20,
                max_generation_steps = 25):
        self.scaling_rate = scaling_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.generation_counter = 0
        self.all_generations = []
        self.max_generation_steps = max_generation_steps
    

    def add_params(self, params):
        for par in params.keys():
            val_range = params[par]["val_range"]
        self.parameters = params


    def init_generation(self):
        self.generation = []
        self.generation_costs = []
        for i in range(self.population_size):
            individual = Individual_Consensus(self.parameters)
            random_state = i*len(self.parameters)
            individual.init_current(random_state)
            individual.train(5)
            self.generation_costs.append(individual.get_accuracy())
    

    def get_trials(self):
        print("--- Trials are computed ---")
        self.trials = []
        self.trials_costs = []
        for i in range(self.population_size):
            target = self.generation[i]
            mutant = self.mutate(i)
            trial = self.crossover(mutant, target)
            trial.train(5)
            self.trials_costs.append(trial.get_accuracy())
            self.trials.append(trial)
    

    def mutate(self, i):
        candidates = self.get_candidates(i)
        mutant = Individual_Consensus(self.generation[candidates[0]].weights.copy())
        for par in self.parameters.keys():
            A = self.generation[candidates[1]].weights[par]["current"]
            B = self.generation[candidates[2]].weights[par]["current"]
            if type(A) is float:
                mutation = mutant.weights[par]["current"] + self.scaling_rate * (A-B)
            else:
                mutation = mutant.weights[par]["current"] + np.int(np.round(self.scaling_rate * (A-B)))
            mutation = self.check_boundaries(mutation,
                                                mutant.weights[par]["val_range"][0],
                                                mutant.weights[par]["val_range"][1])
            mutant.weights[par]["current"] = mutation
        return mutant

    
    def check_boundaries(self, mutation, low_bound, high_bound):
        if mutation < low_bound:
            mutation = low_bound
        elif mutation > high_bound:
            mutation = high_bound
        return mutation
    

    def get_candidates(self, i):
        to_select = list(np.arange(self.population_size))
        to_select.remove(i)
        candidates = np.random.choice(to_select, 3, replace=False)
        return candidates
    

    def crossover(self, mutant, target : Individual_Consensus):
        param_list = list(self.parameters.keys())
        random_par = np.random.choice(param_list)
        trial = Individual_Consensus(target.weights.copy())
        for par in param_list:
            crossover_unit = np.random.uniform(0, 1)
            if crossover_unit <= self.crossover_rate or par == random_par:
                trial.weights[par]["current"] = mutant.weights[par]["current"]
        return trial
    
    def select(self):
        
        idx = np.where(np.array(self.trials_costs) > np.array(self.generation_costs))[0]
        
        # fuse trials and parents to form a candidate pool
        fusion = self.trials_costs[:]
        fusion.extend(self.generation_costs[:])
        candidate_pool = self.trials[:]
        candidate_pool.extend(self.generation[:]) 
        
        # sort score (descending)
        idx = np.argsort(np.array(fusion))[::-1]
    
        # only select the best individuals of both (generation & trials)
        for i in range(len(self.generation)):
            self.generation[i] = candidate_pool[idx[i]]
            self.generation_costs[i] = fusion[idx[i]]
    

    def select_v2(self):
        idx = np.where(np.array(self.trials_costs) > np.array(self.generation_costs))[0]
        for i in idx:
            self.generation[i] = self.trials[i]
            self.generation_costs[i] = self.trials_costs[i]
    

    def evolve(self):
        self.best_solutions = []
        self.worst_solutions = []
        self.init_generation()
        self.all_generations.append(self.generation[:])
        while(self.generation_counter < self.max_generation_steps):
            self.get_trials()
            self.select()
            self.all_generations.append(self.generation[:])
            self.generation_counter += 1
            self.best_solutions.append(np.max(self.generation_costs))
            self.worst_solutions.append(np.min(self.generation_costs))