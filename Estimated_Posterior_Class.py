import numpy as np

import emcee

class Estimated_Posterior(object):    
    def __init__(self):
        self.L = 0
        self.Lslow_Lfast_cond = 0
        self.Lslow_Lfast_marg = 0
        self.Lfast_cond = 0
        self.mean_matrix_fast_cond = 0
        self.L_joint = 0
        self.mean_matrix_fast_cond_joint = 0
        self.PMSS_normalizer = 0
        self.PMSS_inverse = 0
        self.PMSS_normalizer_joint = 0
        self.PMSS_inverse_joint = 0
        self.estimated_covariance = 0
        self.estimated_mean = 0

    def _estimate_mean_cov(self, P, iterations = 20000):
        nwalkers = P.n * 2 * 2     
        p0 = np.zeros((nwalkers, P.n))
        for i in range(nwalkers):
            p0[i,:] = P.mean + np.random.normal(0,0.01,P.n)
        sampler = emcee.EnsembleSampler(nwalkers, P.n, P, args=[])
        sampler.reset()
        sampler.run_mcmc(p0, iterations)
        estimated_covariance = np.cov(sampler.flatchain.T)
        estimated_mean = np.mean(sampler.flatchain.T,1)
        self.estimated_covariance = estimated_covariance
        self.estimated_mean = estimated_mean

    def _compute_matrices(self, P): 
    #This is used for the Metropolis-Hastings method and the Fast-slow decorrelation one
        L = np.linalg.cholesky(self.estimated_covariance)   
        A = self.estimated_covariance[0 : P.number_slow,0:P.number_slow]
        B = self.estimated_covariance[P.number_slow : P.n, P.number_slow : P.n]
        C = self.estimated_covariance[0 : P.number_slow, P.number_slow : P.n]
        cov_slow_cond = A - np.dot(np.dot(C,np.linalg.inv(B)), C.T)
        Lslow_cond =  np.linalg.cholesky(cov_slow_cond)
        cov_fast_cond = B - np.dot(np.dot(C.T,np.linalg.inv(A)), C)
        Lfast_cond =  np.linalg.cholesky(cov_fast_cond)
        mean_matrix_slow_cond = np.dot(C,np.linalg.inv(B))
        mean_matrix_fast_cond = np.dot(C.T,np.linalg.inv(A))   
    # This is used for the Extra updates Metropolis method
        Lslow_Lfast1 = np.concatenate( (Lslow_cond, np.zeros((P.number_slow, P.n - P.number_slow)) ), axis=1)
        Lslow_Lfast2 = np.concatenate( (np.zeros((P.n - P.number_slow, P.number_slow)), Lfast_cond ), axis=1)
        Lslow_Lfast_cond = np.concatenate((Lslow_Lfast1, Lslow_Lfast2), axis=0)
    # This could be used for the Extra updates Metropolis method to update the variables with the marginal instead of the conditional
        Lslow_marg =  np.linalg.cholesky(A)
        Lfast_marg = np.linalg.cholesky(B)
        Lslow_Lfast1b = np.concatenate( (Lslow_marg, np.zeros((P.number_slow, P.n - P.number_slow)) ), axis=1)
        Lslow_Lfast2b = np.concatenate( (np.zeros((P.n - P.number_slow, P.number_slow)), Lfast_marg ), axis=1)
        Lslow_Lfast_marg = np.concatenate((Lslow_Lfast1b, Lslow_Lfast2b), axis=0)
# This is used for the unbiased estimator of the Pseudo-Marginal approach 1
        PMSS_det = np.linalg.det(cov_fast_cond)
        PMSS_normalizer = 1/   (   np.sqrt(PMSS_det)  *  np.sqrt(2 * np.pi) ** (P.n - P.number_slow)     )    
        PMSS_inverse = np.linalg.inv(cov_fast_cond)
# This is used for the unbiased estimator of the Pseudo-Marginal approach 2
        A2 = self.estimated_covariance[0 : P.number_cosmo,0:P.number_cosmo]
        B2 = self.estimated_covariance[P.number_cosmo : P.n, P.number_cosmo : P.n]
        C2 = self.estimated_covariance[0 : P.number_cosmo, P.number_cosmo : P.n]
        cov_fast_cond2 = B2 - np.dot(np.dot(C2.T,np.linalg.inv(A2)), C2)
        L_joint =  np.linalg.cholesky(cov_fast_cond2)    
        mean_matrix_fast_cond_joint = np.dot(C2.T,np.linalg.inv(A2))
        PMSS_det_joint = np.linalg.det(cov_fast_cond2)
        PMSS_normalizer_joint = 1/   (    np.sqrt(PMSS_det_joint)  *  np.sqrt(2 * np.pi) ** (P.n - P.number_slow)     )    
        PMSS_inverse_joint = np.linalg.inv(cov_fast_cond2)
        self.L = L
        self.Lslow_Lfast_cond = Lslow_Lfast_cond
        self.Lslow_Lfast_marg = Lslow_Lfast_marg
        self.Lfast_cond = Lfast_cond
        self.mean_matrix_fast_cond = mean_matrix_fast_cond
        self.L_joint = L_joint
        self.mean_matrix_fast_cond_joint = mean_matrix_fast_cond_joint
        self.PMSS_normalizer = PMSS_normalizer
        self.PMSS_inverse = PMSS_inverse
        self.PMSS_normalizer_joint = PMSS_normalizer_joint
        self.PMSS_inverse_joint = PMSS_inverse_joint

