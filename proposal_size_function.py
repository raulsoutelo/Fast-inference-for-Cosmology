import numpy as np

from unbiased_estimator_functions import log_fhat, log_fhat2
from state_update_function import state_update

def proposal_size(experiment, initial_state, initial_evaluation, u, u2):

    iterations2 = 1000
    proposals_size_array = 0.35 * np.ones(experiment.number_updates)
    learning_rate = 0.1 * np.ones(experiment.number_updates)
    initial_proposals_size_array = 0.35 * np.ones(experiment.number_updates)
    previous_prob_u = 1
    initial_u = u
    initial_u2 = u2

    while True:     
        random_number_counter = np.zeros(experiment.number_updates)
        accepted_counter = np.zeros(experiment.number_updates)
        state = initial_state
        previous_evaluation = initial_evaluation
        u = initial_u
        u2 = initial_u2

        if experiment.method == 'MH' or experiment.method == 'extra MH' or experiment.method == 'Lewis':
            iterations2 = 1000
            for i in range(iterations2): 
                random_number = np.random.randint(0, experiment.number_updates)   # if not working well, we could set the real ratio 
                proposal, update_cost = state_update(experiment, state, proposals_size_array[random_number], random_number)
                evaluation = np.exp(experiment.posterior(proposal))
                if np.random.rand() < (evaluation / previous_evaluation):
                    state = proposal
                    previous_evaluation = evaluation
                    accepted_counter[random_number] = accepted_counter[random_number] + 1
                random_number_counter[random_number] = random_number_counter[random_number] + 1

        elif experiment.method == 'APM1' or experiment.method == 'APM2':  
            iterations2 = 10000
            look_for_new_u = True
            for i in range(iterations2):
	        if experiment.method == 'APM1':
                    random_number = np.random.randint(0, 1 + experiment.posterior.relative_speed/experiment.number_ensembles + 1)
		else:
		    random_number = np.random.randint(0, 2)
                if random_number > 1:
                    random_number = 1     
                if random_number == 0:    # we update the state with Metropolis-Hastings
                    proposal, update_cost = state_update(experiment, state, proposals_size_array[random_number], random_number)   
	            if experiment.method == 'APM1':
                        evaluation = np.exp(log_fhat(experiment, proposal[0 : experiment.posterior.number_slow], u))       
                    else:
			evaluation = np.exp(log_fhat2(experiment, proposal[0 : experiment.posterior.number_slow], u, u2)) 
                    if np.random.rand() < (evaluation / previous_evaluation):
                        state = proposal
                        previous_evaluation = evaluation
                        accepted_counter[random_number] = accepted_counter[random_number] + 1
                        look_for_new_u = True   
                else:               # we update u with metropolis independent proposals   
                    if experiment.fast_update == 'independent':
                        if look_for_new_u:
                            proposed_u = np.random.normal(0,1,(experiment.number_ensembles, experiment.posterior.n - experiment.posterior.number_slow))
			    proposed_u2 = np.random.normal(0,1,(experiment.posterior.number_slow - experiment.posterior.number_cosmo))   
                            if experiment.method == 'APM1':
                                evaluation = np.exp(log_fhat(experiment, state[0 : experiment.posterior.number_slow], proposed_u))       
                            else:
			        evaluation = np.exp(log_fhat2(experiment, state[0: experiment.posterior.number_slow], proposed_u, proposed_u2)) 
                            if np.random.rand() < (evaluation / previous_evaluation):
                                u = proposed_u
                                u2 = proposed_u2
                                previous_evaluation = evaluation
                                accepted_counter[random_number] = accepted_counter[random_number] + 1
                                look_for_new_u = False
                        else: 
                            random_number_counter[random_number] = random_number_counter[random_number] - 1
                    else:   # we could also update u with Metropolis-Hastings  
			print 'variant not implemented!'
                random_number_counter[random_number] = random_number_counter[random_number] + 1

	else:
            print 'method not implemented'
            break

        percentage = accepted_counter / random_number_counter
        repeat = False
        for i in range(experiment.number_updates):    
            if (experiment.method == 'APM1' or experiment.method == 'APM2') and experiment.fast_update == 'independent' and i==1:  
                break     
            if percentage[i] > 0.3 or percentage[i] < 0.2:
                proposals_size_array[i] = proposals_size_array[i] + learning_rate[i]*(percentage[i] - 0.234)
                if proposals_size_array[i] < 0:
                    learning_rate[i] = learning_rate[i] / 3
                    proposals_size_array[i] = 0.00000001
                repeat = True

        if repeat == False:
            string_of_percentage = ''
            string_of_proposals_size = ''
            for i in range(experiment.number_updates): 
                string_of_percentage = string_of_percentage + ' ' + str(percentage[i])
                string_of_proposals_size = string_of_proposals_size + ' ' +  str(proposals_size_array[i])
            print 'preliminary chain has finished, percentage: ' + string_of_percentage + ' and proposolas_size: ' + string_of_proposals_size
            break

    return proposals_size_array
