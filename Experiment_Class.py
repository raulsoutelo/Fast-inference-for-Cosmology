import numpy as np

class Experiment:
    def __init__(self, method, main_update, fast_update, P, Est_P, number_ensembles = 1):
        self.method = method
        self.main_update = main_update
        self.fast_update = fast_update
        self.number_ensembles = number_ensembles
        self.posterior = P
        self.est_posterior = Est_P
        self.sample = [] 
        self.mean_cost = 0
        self.time = 0 
	self.total_cost = 0
        self.acceptance_rates = np.zeros(2)
        self.number_updates = 2
        if self.method == 'MH':
            self.number_updates = 1


