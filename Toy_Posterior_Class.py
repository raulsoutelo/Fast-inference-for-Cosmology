import numpy as np
import os
import sys
from scipy import linalg

class Posterior(object):
    def __init__(self):
        if True:
	    # WE CREATE EITHER A RANDOM MULTIVARIATE GAUSSIAN OR SET THE VALUES BY HAND 
            slow_dim = 2
	    fast_dim = 2
            np.random.seed(0)
	    if True:
    	        A = np.random.normal(0,1,(slow_dim+fast_dim,slow_dim+fast_dim))
    	        covariance = np.dot(A,A.transpose())
                slow_mean = np.random.normal(0,1,slow_dim)
                fast_mean = np.random.normal(0,1,fast_dim)
                slow_min = slow_mean - 8*np.random.rand(slow_dim)
                slow_max = slow_mean + 8*np.random.rand(slow_dim)   
                fast_min = fast_mean - 8*np.random.rand(fast_dim)
                fast_max = fast_mean + 8*np.random.rand(fast_dim)           
	    else:
                slow_dim = 2
                fast_dim = 2
                slow_mean = np.array([1 , 6])
                slow_min = np.array([0.5, 1.5])
                slow_max = np.array([5.3, 6.4])
                fast_mean = np.array([2, 5])
                covariance = np.array([(1.,0.6,0.5,0.4), (0.6,1.3,0.3,0.4), (0.5,0.3,1.1,0.7), (0.4,0.4,0.7,1.1)])

            # WE CHECK IF THE COVARIANCE MATRIX IS SEMIDEFINITE POSITIVE
            def isPSD(covariance, tol=1e-8):
                E,V = linalg.eigh(covariance)
                return np.all(E > -tol)

            if isPSD(covariance):
                print 'right covariance'
            else:
                print 'wrong covariance'
	
            self.covmat = covariance
            self.inv = np.linalg.inv(self.covmat)
            self.mean = np.concatenate((slow_mean,fast_mean),axis=0)
            self.lower = np.concatenate((slow_min,fast_min),axis=0)
            self.upper = np.concatenate((slow_max,fast_max),axis=0)
            self.n = len(self.mean)
	    self.number_slow = slow_dim

    def __call__(self, x):
        assert len(x)==self.n, "Wrong length vector passed in - should be {} but was {}".format(self.n,len(x))
        x = np.array(x)
        if np.any(x<self.lower) or np.any(x>self.upper):
            return -np.inf
        d = x - self.mean
        chi2 = np.dot(np.dot(d,self.inv),d)
        return -0.5 * chi2

class Toy_Posterior(Posterior):
    def __init__(self):
        super(Toy_Posterior, self).__init__()
        self.cost = np.zeros(self.n)
        self.number_cosmo = self.number_slow				# for this particular toy posterior
        self.slow_mask = np.zeros(self.n)
        for i in range(self.n):
	    if i < self.number_slow:
		self.cost[i] = 20
                self.slow_mask[i] = 1
	    else:
		self.cost[i] = 0.01
        self.fast_mask = np.ones(self.n) - self.slow_mask
        self.relative_speed = np.amax(self.cost)/np.min(self.cost) 
        
    def _calculate_cost(self, modified_variables):  	# the cost corresponds to the maximum cost of the variables modified
        assert len(modified_variables)==self.n, "Wrong length vector passed in - should be {} but was {}".format(self.n,len(modified_variables))
        total_cost = 0
        for i in range(self.n):
	    if modified_variables[i] == 1 and self.cost[i] > total_cost:
		total_cost = self.cost[i]
        return total_cost

    def _sort_by_cost(self):
	return None

