import numpy as np

# We propose to move some of the variables
def metropolis_propose(state, mask, proposals_size):  
    return mask * proposals_size * np.random.normal(0,1,len(state))	

# We propose the new state and evaluate the cost of the update
def state_update(experiment, state, proposals_size, random_number):
    current_cost = 0
    proposal = state
    movement = np.zeros(experiment.posterior.n)              

    if experiment.method == 'MH':  						
        modified_variables = np.ones(experiment.posterior.n) 
        mask = np.ones(experiment.posterior.n)
        movement = metropolis_propose(state, mask, proposals_size)
        proposal = state + np.dot(experiment.est_posterior.L,movement)

    elif experiment.method == 'Lewis': 	
        if random_number == 0:
            modified_variables = np.ones(experiment.posterior.n) 
            mask = experiment.posterior.slow_mask
            movement = metropolis_propose(state, mask, proposals_size)
        else:
            modified_variables = experiment.posterior.fast_mask 
            mask = experiment.posterior.fast_mask
            movement = metropolis_propose(state, mask, proposals_size)	    	    
        proposal = state + np.dot(experiment.est_posterior.L,movement) 

    elif experiment.method == 'extra MH' or experiment.method == 'APM1' or experiment.method == 'APM2':
        if (experiment.main_update == 'marg' and experiment.fast_update == 'marg') or experiment.method == 'APM1' or experiment.method == 'APM2':
            L_temporal = experiment.est_posterior.Lslow_Lfast_marg
        elif experiment.main_update == 'cond' and experiment.fast_update == 'cond':
            L_temporal = experiment.est_posterior.Lslow_Lfast_cond
        else:
            print 'other method not implemented!'            
        if random_number == 0:
            modified_variables = experiment.posterior.slow_mask
            mask = experiment.posterior.slow_mask
            movement = metropolis_propose(state, mask, proposals_size)
        else:
            modified_variables = experiment.posterior.fast_mask 
            mask = experiment.posterior.fast_mask
            movement = metropolis_propose(state, mask, proposals_size)
        proposal = state + np.dot(L_temporal, movement)

    else:
        print 'other method not implemented!'

    current_cost = experiment.posterior._calculate_cost(modified_variables)
    return proposal, current_cost


