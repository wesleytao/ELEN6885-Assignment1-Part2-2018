import numpy as np
from RLalgs.utils import action_evaluation

def value_iteration(env, gamma, max_iteration, theta):
    """
    Implement value iteration algorithm. 

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
            Number of iterations
    """

    V = np.zeros(env.nS)
    numIterations = 0

    #Implement the loop part here
    ############################
    # YOUR CODE STARTS HERE
    max_diff = 0 
    while(numIterations ==0 or max_diff > theta):
        max_diff = 0
        numIterations = numIterations + 1
        for this_state in range(env.nS):
            opt_action = -1 
            opt_value = -1 
            for this_action in range(env.nA):
                action_value = sum([p*(r+V[s]*gamma) for p,s,r,t in env.P[this_state][this_action]])
                if action_value > opt_value:
                    opt_action = this_action
                    opt_value = action_value
            # end of value 
            if abs(V[this_state] - opt_value)> max_diff:
                max_diff = abs(V[this_state] - opt_value)
            V[this_state] = opt_value
   
    
    
    # YOUR CODE ENDS HERE
    ############################
    
    #Extract the "optimal" policy from the value function
    policy = extract_policy(env, V, gamma)
    
    return V, policy, numIterations

def extract_policy(env, v, gamma):

    """ 
    Extract the optimal policy given the optimal value-function.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    v: numpy.ndarray
        value function
    gamma: float
        Discount factor. Number in range [0, 1)
    
    Outputs:
    policy: numpy.ndarray
    """

    policy = np.zeros(env.nS, dtype = np.int32)
    ############################
    # YOUR CODE STARTS HERE
    for this_state in range(env.nS):
        opt_action = -1
        max_v = -1
        for this_action in range(env.nA):
            this_value = sum([p*(r+gamma*v[s])  for p,s,r,t in env.P[this_state][this_action]])
            if this_value > max_v:
                max_v = this_value
                opt_action = this_action
        policy[this_state] = opt_action
    # YOUR CODE ENDS HERE
    ############################

    return policy