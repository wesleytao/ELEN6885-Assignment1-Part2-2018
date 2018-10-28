import numpy as np
from RLalgs.utils import action_evaluation

def policy_iteration(env, gamma, max_iteration, theta):
    """
    Implement Policy iteration algorithm.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
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
    """

    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype = np.int32)
    policy_stable = False
    numIterations = 0
    
    while not policy_stable and numIterations < max_iteration:
        #Implement it with function policy_evaluation and policy_improvement
        ############################
        # YOUR CODE STARTS HERE
        V = policy_evaluation(env,policy,gamma,theta)
        policy, policy_stable      = policy_improvement(env,V, policy,gamma)

        # YOUR CODE ENDS HERE
        ############################
        numIterations += 1
        
    return V, policy, numIterations


def policy_evaluation(env, policy, gamma, theta):
    """
    Evaluate the value function from a given policy.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    gamma: float
            Discount factor.
    policy: numpy.ndarray
            The policy to evaluate. Maps states to actions.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
            The value function from the given policy.
    """
    ############################
    # YOUR CODE STARTS HERE
    assert(np.all(np.isin(policy,range(env.nA)))) # check all policy are within the action space
    V = np.zeros(env.nS) # init state values
    max_diff = 0
    n_iter = 0
    temp_state = -1
    while(n_iter ==0  or max_diff > theta):
        n_iter = n_iter + 1
        max_diff = 0
        for s in range(env.nS):
            new_v = sum([p*(r+gamma*V[s]) for p,s,r,t in env.P[s][policy[s]]])
            if abs(new_v-V[s]) > max_diff:
                temp_state = s
                max_diff = abs(new_v - V[s])
            V[s] = new_v
        if n_iter > 10000:
            print("reached 10000 iterations and failed to converge")
            break
    # YOUR CODE ENDS HERE
    ############################

    return V


def policy_improvement(env, value_from_policy, policy, gamma):
    """
    Given the value function from policy, improve the policy.

    Inputs:
    env: OpenAI Gym environment
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

    value_from_policy: numpy.ndarray
            The value calculated from the policy
    policy: numpy.ndarray
            The previous policy.
    gamma: float
            Discount factor.

    Outputs:
    new policy: numpy.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    policy_stable: boolean
            True if the "optimal" policy is found, otherwise false
    """
    ############################
    # YOUR CODE STARTS HERE
    q = np.zeros((env.nS,env.nA))
    for this_state in range(env.nS):
        for this_action in range(env.nA):
            q[this_state][this_action] =sum([p*(r+gamma*value_from_policy[s])  for p,s,r,t in env.P[this_state][this_action]])
    new_policy = np.argmax(q,1)
            
    if np.all(policy == new_policy):
        policy_stable = True
    else:
        policy_stable = False
    policy = new_policy

    # YOUR CODE ENDS HERE
    ############################

    return policy, policy_stable