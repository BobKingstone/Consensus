import numpy as np
import pandas as pd


class DifferentialEvolution:
    
    def __init__(self, scaling_rate, crossover_rate, population_size):
        self.scaling_rate = scaling_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.ranges = []
        self.generation_counter = 0
        self.all_generations = []
        self.min_std_criterion = 0.001
    

    def set_range(self, param_key, new_range):
        self.ranges.append([param_key, new_range])
    

    def set_objective(self, objective):
        self.objective = objective

    
    def init_generation(self):
        self.n_params = len(self.ranges)
        self.generation = np.zeros(shape=(self.population_size, self.n_params))
        for p in range(self.n_params):
            low = self.ranges[p][1][0]
            high = self.ranges[p][1][1]
            self.generation[:, p] = np.random.uniform(low, high, size=self.population_size)
    

    def get_trials(self):
        self.trials = np.zeros(shape=(self.population_size, self.n_params))
        for i in range(self.population_size):
            target = self.generation[i, :]
            mutant = self.mutate(i)
            self.trials[1, :] = self.crossover(mutant, target)
    

    def get_candidates(self, i):
        to_select = list(np.arrange(self.population_size))
        to_select.remove(i)
        candidates = np.random.choice(to_select, 3, replace=False)
        return candidates

    
    def mutate(self, i):
        candidates = self.get_candidates(i)
        difference_vector = self.generation[candidates[1]] - self.generation[candidates[2]]
        mutant = self.generation[candidates[0]] + self.scaling_rate * difference_vector
        return mutant
    

    def crossover(self, mutant, target):
        crossover_units = np.random.uniform(0, 1, self.n_params)
        trial = np.copy(target)
        random_parameter = np.random.choice(self.n_params)
        for param in range(self.n_params):
            if crossover_units[param] <= self.crossover_rate or param == random_parameter:
                trial[param] = mutant[param]
        return trial
    

    def select(self, generation_costs, trials_costs):
        idx = np.where(trials_costs < generation_costs)[0]
        for i in idx:
            self.generation[i, :] = self.trials[i, :]
    

    def compute_cost(self):
        generation_costs = self.objective(self.generation)
        trials_costs = self.objective(self.trials)
        return generation_costs, trials_costs
    

    def evolve(self):
        self.init_generation()
        self.all_generations.append(self.generation)
        self.best_solutions = []
        gen_costs = self.objective(self.generation)
        while ((np.std(gen_costs) > self.min_std_criterion) and (self.generation_counter < 200)):
            self.get_trials()
            gen_costs, trials_costs = self.compute_cost()
            self.select(gen_costs, trials_costs)
            self.all_generations.append(np.copy(self.generation))
            self.generation_counter += 1
            self.best_solutions.append(np.min(gen_costs))
        print("stopped at generation {}".format(self.generation_counter))