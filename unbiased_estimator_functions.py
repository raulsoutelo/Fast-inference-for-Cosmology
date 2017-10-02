import numpy as np
import scipy
from scipy.misc import logsumexp

# Unbiased estimator for Pseudo-Marginal approach 1           
def log_fhat(experiment, theta_slow, u):
    output = np.zeros(experiment.number_ensembles) 
    q = np.zeros(experiment.number_ensembles)
    current_output = 0
    for i in range(experiment.number_ensembles):
        theta_fast = np.dot(experiment.est_posterior.Lfast_cond,  u[i , :])
        current_q = experiment.est_posterior.PMSS_normalizer * np.exp( -0.5 * np.dot(np.dot(theta_fast, experiment.est_posterior.PMSS_inverse), theta_fast))
        theta_fast = theta_fast + experiment.est_posterior.estimated_mean[experiment.posterior.number_slow:experiment.posterior.n] + np.dot(experiment.est_posterior.mean_matrix_fast_cond,theta_slow - experiment.est_posterior.estimated_mean[0:experiment.posterior.number_slow])  
        output[i] = experiment.posterior( np.concatenate((theta_slow, theta_fast),axis=0) ) - np.log(current_q)
    return logsumexp(output)

# Unbiased estimator for Pseudo-Marginal approach 2
def log_fhat2(experiment, theta_slow, u, u2):     
    output = np.zeros(experiment.number_ensembles) 
    q = np.zeros(experiment.number_ensembles)
    for i in range(experiment.number_ensembles):
	theta_nuisance = np.dot(experiment.est_posterior.L_joint, np.concatenate((u2, u[i , :]),axis=0))
        current_q = experiment.est_posterior.PMSS_normalizer_joint * np.exp( -0.5 * np.dot(np.dot(theta_nuisance, experiment.est_posterior.PMSS_inverse_joint), theta_nuisance))
        theta_nuisance = theta_nuisance + experiment.est_posterior.estimated_mean[experiment.posterior.number_cosmo:experiment.posterior.n] + np.dot(experiment.est_posterior.mean_matrix_fast_cond_joint, theta_slow[0:experiment.posterior.number_cosmo] - experiment.est_posterior.estimated_mean[0:experiment.posterior.number_cosmo])
        output[i] = experiment.posterior( np.concatenate((theta_slow[0:experiment.posterior.number_cosmo], theta_nuisance),axis=0) ) - np.log(current_q)
    return logsumexp(output)




