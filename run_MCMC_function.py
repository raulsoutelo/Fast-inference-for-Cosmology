import numpy as np
from datetime import datetime

from proposal_size_function import proposal_size
from unbiased_estimator_functions import log_fhat, log_fhat2
from state_update_function import state_update
 
def convergence(samples, experiment):
    threshold = 0.1
    for j in range(experiment.posterior.number_cosmo):
        x = np.array(samples).T[j]        
        for i in range(int(len(x)/4), int(len(x)/2)):
            correlation = np.corrcoef(np.array([x[0:len(x)-i], x[i:len(x)]]))[0,1]
            if correlation > threshold or correlation < -threshold:
                print 'the autocorrelation is ' + str(correlation) + ' for a percentage of  ' + str(float(i - int(len(x)/4))/(  int(len(x)/2) - int(len(x)/4)  ) ) + ' of variable ' + str(j)
                return False
    return True 	    

def run_MCMC(experiment):

    t0 = datetime.now() 
    initial_time = datetime.now()
    rate_convergence = 600                  # we check if the chain has converged every rate_convergence seconds
    cost_since_lastcheck = 0                # we also check each 1M of seconds in real time
    rate_costchecking = 1000000             # we also check each 1M of seconds in real time
    maximum_length = 3600 * 24 * 365
    samples = []
    acc_cost = 0
    number_appends = 0
    current_cost = 0

# Initialization of the MCMC chain
    initial_state = experiment.posterior.mean
    initial_evaluation = np.exp(experiment.posterior(initial_state))
    u = np.zeros((experiment.number_ensembles, experiment.posterior.n - experiment.posterior.number_slow))
    u2 = np.zeros( experiment.posterior.number_slow - experiment.posterior.number_cosmo)
    previous_prob_u = 1
    previous_prob_theta = 1
    look_for_new_u = True
    # Maybe the previous value of the initial evaluation is also fine for the Pseudo-Marginal approaches
    if experiment.method == 'APM1':
        initial_evaluation = np.exp(log_fhat(experiment, initial_state[0 : experiment.posterior.number_slow], u))
    elif experiment.method == 'APM2':
        initial_evaluation = np.exp(log_fhat2(experiment, initial_state[0 : experiment.posterior.number_slow], u, u2))               
           
# We tune the proposals size
    proposals_size = proposal_size(experiment, initial_state, initial_evaluation, u, u2)

# We run the proper chain
    state = initial_state[:]
    previous_evaluation = initial_evaluation
    random_number_counter = np.zeros(experiment.number_updates)
    accepted_counter = np.zeros(experiment.number_updates)
    i = 0
    while True:
        if experiment.method == 'MH' or experiment.method == 'extra MH' or experiment.method == 'Lewis':
            random_number = np.random.randint(0, 1 + experiment.posterior.relative_speed*(experiment.number_updates - 1))
        else:
	    if experiment.method == 'APM1':
                random_number = np.random.randint(0, 1 + experiment.posterior.relative_speed/experiment.number_ensembles + 1)
	    else:
                random_number = np.random.randint(0, 2)
        if random_number > 1:
            random_number = 1

        if experiment.method == 'MH' or experiment.method == 'extra MH' or experiment.method == 'Lewis':
            proposal, update_cost = state_update(experiment, state, proposals_size[random_number], random_number)
            evaluation = np.exp(experiment.posterior(proposal))
            if np.random.rand() < (evaluation / previous_evaluation):
                state = proposal
                previous_evaluation = evaluation
                accepted_counter[random_number] = accepted_counter[random_number] + 1
                if random_number == 0:
                    samples.append(state[0:experiment.posterior.number_cosmo])  
                    acc_cost = acc_cost + current_cost
                    cost_since_lastcheck = cost_since_lastcheck + current_cost 	
                    number_appends = number_appends + 1	 
                    current_cost = 0 

        elif experiment.method == 'APM1' or experiment.method == 'APM2':            
            if random_number == 0: 
                proposal, update_cost = state_update(experiment, state, proposals_size[random_number], random_number)   
                update_cost = update_cost + (experiment.number_ensembles - 1) * experiment.posterior._calculate_cost(experiment.posterior.fast_mask) 
	        if experiment.method == 'APM1':
                    evaluation = np.exp(log_fhat(experiment, proposal[0 : experiment.posterior.number_slow], u))       
                else:
		    evaluation = np.exp(log_fhat2(experiment, proposal[0 : experiment.posterior.number_slow], u, u2)) 
                if np.random.rand() < (evaluation / previous_evaluation):
                    state = proposal
                    previous_evaluation = evaluation
                    accepted_counter[random_number] = accepted_counter[random_number] + 1
                    samples.append(state[0:experiment.posterior.number_cosmo])  
                    acc_cost = acc_cost + current_cost
                    cost_since_lastcheck = cost_since_lastcheck + current_cost 	
                    number_appends = number_appends + 1	 
                    current_cost = 0
                    look_for_new_u = True
            else:                                    # we update u with metropolis independent proposals   
                if experiment.fast_update == 'independent':
		    if experiment.method == 'APM1':
		        update_cost = (experiment.number_ensembles) * experiment.posterior._calculate_cost(experiment.posterior.fast_mask) 
		    else:
	                update_cost = experiment.posterior._calculate_cost(experiment.posterior.slow_mask)
	                update_cost = update_cost + (experiment.number_ensembles - 1) * experiment.posterior._calculate_cost(experiment.posterior.fast_mask)    
                    if look_for_new_u:
                        proposed_u = np.random.normal(0,1,(experiment.number_ensembles, experiment.posterior.n - experiment.posterior.number_slow))
			proposed_u2 = np.random.normal(0,1,(experiment.posterior.number_slow - experiment.posterior.number_cosmo))   
	                if experiment.method == 'APM1':
                            evaluation = np.exp(log_fhat(experiment, state[0 : experiment.posterior.number_slow], proposed_u))       
                        else:
			    evaluation = np.exp(log_fhat2(experiment, state[0 : experiment.posterior.number_slow], proposed_u, proposed_u2)) 
                        if np.random.rand() < (evaluation) / (previous_evaluation):
                            u = proposed_u
                            u2 = proposed_u2
                            previous_evaluation = evaluation
                            accepted_counter[random_number] = accepted_counter[random_number] + 1
                            look_for_new_u = False
                    else: 
                        random_number_counter[random_number] = random_number_counter[random_number] - 1

	else:
            print 'method not implemented'
            break               

        random_number_counter[random_number] = random_number_counter[random_number] + 1
        current_cost = current_cost + update_cost
        i += 1
        
# We check if the MCMC has converged
        t1 = datetime.now()	
        if ((t1 - t0).total_seconds() > rate_convergence or cost_since_lastcheck > rate_costchecking) and len(samples) > 10:
            convergence_achieved = convergence(samples, experiment)
            t0 = datetime.now()
            cost_since_lastcheck = 0
            percentage = accepted_counter / random_number_counter
            print 'percentage of acceptance is: ' + str(percentage)
            print 'total seconds in calculating correlation is ' + str((datetime.now() - t1).total_seconds()) + ' and number of iterations ' + str(i)
            print 'the total cost involved is ' + str(acc_cost)
            if convergence_achieved:
                break

# We check if we have achieved the maximum cost to stop the MCMC
        if acc_cost > maximum_length:
            print 'THE CHAIN WAS NOT ABLE TO CONVERGE IN ONE YEAR TIME!!!'
            break
    
# We print the percentage of acceptance and maximum idealized cost
    percentage = accepted_counter / random_number_counter
    string_of_percentage = ''
    for i in range(experiment.number_updates): 
        string_of_percentage = string_of_percentage + ' ' + str(percentage[i])
    print 'proper chain has finished, percentage: ' + string_of_percentage
    print 'the total idealized cost is: ' + str(acc_cost)

# We add the data to the experiment
    cost = acc_cost / number_appends
    samples2 = np.array(samples).T
    total_time =  (datetime.now() - initial_time).total_seconds()     
    experiment.sample = samples2
    experiment.mean_cost = cost
    experiment.time = total_time
    experiment.total_cost = acc_cost
    experiment.acceptance_rates = percentage

    return experiment
