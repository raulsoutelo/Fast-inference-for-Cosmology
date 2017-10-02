import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot(experiments, delay, histogram = True, autocorrelation = True, trace = True):

    # WE PLOT THE HISTOGRAMS TO CHECK THAT THE DISTRIBUTION WAS EXPLORED PROPERLY
    	# The visualization should match if:
    	# 1) There is no slow variables
    	# 2) In the slow variable if there is just one
    	# 3) In every variable if deleted_covariance is set to True  
    for k, v in experiments.items():
        cosmological_variables = v.posterior.number_cosmo
    plot_index = 0
    if histogram:
        for k, v in experiments.items(): 
	    for i in range (v.posterior.number_cosmo):
                plot_index = plot_index + 1
        	plt.figure(plot_index)
        	count, bins, ignored = plt.hist(v.sample[i], 100, normed=True)
		low = norm.cdf(v.posterior.lower[i],v.posterior.mean[i],np.sqrt(v.posterior.covmat[i,i]))
        	high = 1 - norm.cdf(v.posterior.upper[i],v.posterior.mean[i],np.sqrt(v.posterior.covmat[i,i]))
        	plt.plot(bins, 1/(np.sqrt(v.posterior.covmat[i,i]) * np.sqrt(2 * np.pi) * (1-low-high)) * np.exp( - (bins - v.posterior.mean[i])**2 / (2 * v.posterior.covmat[i,i]) ),linewidth=2, color='r', label= str(k) + ': theta' + str(i+1))
		plt.legend()		

    if trace:
        for k, v in experiments.items():
	    for i in range (v.posterior.number_cosmo):
                plt.xlabel('idealized cost (seconds)', fontsize=14)
        	plot_index = plot_index + 1
        	plt.figure(plot_index)
                cost_per_iteration =  range(len(v.sample[i]))
                for j in range(len(v.sample[i])):
		    cost_per_iteration[j] = cost_per_iteration[j]*v.mean_cost
                plt.plot(cost_per_iteration, v.sample[i], label= str(k) + ': theta' + str(i+1))
		plt.legend()	

    if autocorrelation: 
	def autocorr(x):
	    for i in range(0,correlation_points):
		delay = int(np.round(step*i/mean_cost))  #delay in iterations, longer steps when the mean cost is lower in order to match later!
		correlation[i] = np.corrcoef(np.array([x[0:len(x)-delay], x[delay:len(x)]]))[0,1]
		lag[i] = int(delay*mean_cost)  #we multiply by the average cost of each iteration in order to compare methods! 
	    return correlation, lag
	correlation_points = 1000
	for i in range (cosmological_variables):
	    plot_index = plot_index + 1
	    plt.figure(plot_index)
	    plt.grid()
            initialize = True
            for k, v in experiments.items():
		if initialize:
		    maximum_delay = (len(v.sample[i]) * v.mean_cost)/2
		    initialize = False
		if (len(v.sample[i]) * v.mean_cost)/2 < maximum_delay:
		    maximum_delay = (len(v.sample[i]) * v.mean_cost)/2		    
            if delay > maximum_delay:
		print 'the delay selected is too big! we will have a small vector to calculate de autocorrelation'
	    step = delay / correlation_points
            for k, v in experiments.items():
		correlation = np.zeros(correlation_points)
		lag = np.zeros(correlation_points)
		mean_cost = v.mean_cost
		correlation, lag = autocorr(v.sample[i])
                plt.xlabel('idealized cost (seconds)', fontsize=14)
		plt.plot(lag, correlation, label= str(k) + ': theta' + str(i+1))
	        plt.legend()

    plt.show()

