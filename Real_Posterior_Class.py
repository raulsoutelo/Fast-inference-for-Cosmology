import numpy as np
import os
import sys
from scipy import random, linalg

# This is a hack to get the directory the directory this 
# file is in
DIR = os.path.split(os.path.abspath(__file__))[0]
COVMAT = os.path.join(DIR,"covmat.txt")
MEANS = os.path.join(DIR,"means.txt")
COSTS = os.path.join(DIR,"costs.txt")
COSMO = os.path.join(DIR,"cosmo_parameters.txt")
LIMITS = os.path.join(DIR,"limits.txt")


class Posterior(object):
    def __init__(self):
        self.covmat = np.loadtxt(COVMAT)
        self.inv = np.linalg.inv(self.covmat)
        self.mean = np.loadtxt(MEANS)
        self.lower,self.upper = np.loadtxt(LIMITS).T
        self.n = len(self.mean)

    def __call__(self, x):
        assert len(x)==self.n, "Wrong length vector passed in - should be {} but was {}".format(self.n,len(x))
        x = np.array(x)
        if np.any(x<self.lower) or np.any(x>self.upper):
            return -np.inf
        d = x - self.mean
        chi2 = np.dot(np.dot(d,self.inv),d)
        return -0.5 * chi2

class Real_Posterior(Posterior):
    def __init__(self):
        super(Real_Posterior, self).__init__()
        self.cost = np.loadtxt(COSTS)
        self.cosmo = np.loadtxt(COSMO)
        self.number_cosmo = 0
        for i in range(self.n):
	    if self.cosmo[i] == 1:
		self.number_cosmo = self.number_cosmo + 1
        self.relative_speed = np.amax(self.cost)/np.min(self.cost) 
        self.number_slow = 0
        self.slow_mask = np.zeros(self.n)
        for i in range(self.n):
	    if self.cost[i] > (np.amax(self.cost) + np.min(self.cost))/2:
		self.number_slow = self.number_slow + 1
                self.slow_mask[i] = 1
        self.fast_mask = np.ones(self.n) - self.slow_mask
        
    def _calculate_cost(self, modified_variables):  
        assert len(modified_variables)==self.n, "Wrong length vector passed in - should be {} but was {}".format(self.n,len(modified_variables))
        total_cost = 0
        for i in range(self.n):
	    if modified_variables[i] == 1 and self.cost[i] > total_cost:
		total_cost = self.cost[i]
        return total_cost

    def _sort_covariance_cost(self, covariance):	
	covariance[:,[6, 19]] = covariance[:,[19, 6]]
        covariance[[6, 19],:] = covariance[[19, 6],:]
	covariance[:,[7, 20]] = covariance[:,[20, 7]]
        covariance[[7, 20],:] = covariance[[20, 7],:]
	covariance[:,[8, 21]] = covariance[:,[21, 8]]
        covariance[[8, 21],:] = covariance[[21, 8],:]	
	covariance[:,[9, 22]] = covariance[:,[22, 9]]
        covariance[[9, 22],:] = covariance[[22, 9],:]
	return covariance

    def _sort_array_cost(self, array):		
	array[[6, 19]] = array[[19, 6]]
	array[[7, 20]] = array[[20, 7]]
	array[[8, 21]] = array[[21, 8]]
	array[[9, 22]] = array[[22, 9]]	
	return array

    def _sort_by_cost(self):
	self.covmat = self._sort_covariance_cost(self.covmat)
	self.mean = self._sort_array_cost(self.mean)
	self.lower = self._sort_array_cost(self.lower)
	self.upper = self._sort_array_cost(self.upper)
        self.slow_mask = self._sort_array_cost(self.slow_mask)
        self.fast_mask = np.ones(self.n) - self.slow_mask
        self.cosmo = self._sort_array_cost(self.cosmo)
        self.cost = self._sort_array_cost(self.cost)
	return None

