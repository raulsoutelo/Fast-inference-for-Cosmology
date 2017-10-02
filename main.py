from Real_Posterior_Class import Real_Posterior
from Toy_Posterior_Class import Toy_Posterior
from Estimated_Posterior_Class import Estimated_Posterior
from Experiment_Class import Experiment
from run_MCMC_function import run_MCMC
from plot_function import plot

# We first create the posterior distribution to be explored
P = Real_Posterior()
P._sort_by_cost()

# We then obtain an estimate of the posterior distribution to be used in all the methods
Est_P = Estimated_Posterior()
Est_P._estimate_mean_cov(P, iterations = 20000)
Est_P._compute_matrices(P) 

# We setup the experiments to be done
experiments = {}
experiments['Metropolis-Hastings'] = Experiment(method = 'MH', main_update = '-', fast_update = '-', P = P, Est_P = Est_P) 
experiments['Extra update Metropolis (C)'] = Experiment(method = 'extra MH', main_update = 'cond', fast_update = 'cond', P = P, Est_P = Est_P) 
experiments['Extra update Metropolis (M)'] = Experiment(method = 'extra MH', main_update = 'marg', fast_update = 'marg', P = P, Est_P = Est_P) 
experiments['Fast-slow decorrelation'] = Experiment(method = 'Lewis', main_update = '-', fast_update = '-', P = P, Est_P = Est_P)
experiments['APM1 MI+MH'] = Experiment(method = 'APM1', main_update='-', fast_update='independent', P=P, Est_P = Est_P, number_ensembles = 10)
experiments['APM2 MI+MH'] = Experiment(method = 'APM2', main_update='-', fast_update='independent', P = P, Est_P = Est_P, number_ensembles = 10)

# we run the experiments and plot the comparison of the different methods
def main():
    for k, v in experiments.items(): 
	print 'running ' + str(k)
	v = run_MCMC(experiment = v)
        print 'method ' + str(k) + ' had an idealized cost: ' + str(v.total_cost) + ' and percentage of acceptance: ' + str(v.acceptance_rates)
    plot(experiments, delay = 10000, histogram = False, autocorrelation = True, trace = False)

if __name__ == '__main__':
    main()
