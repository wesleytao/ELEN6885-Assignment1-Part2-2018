import numpy as np  
from time import sleep

def estimate(OldEstimate, StepSize, Target):
    'An incremental implementation of average.'
    
    NewEstimate = OldEstimate + StepSize * (Target - OldEstimate)
    return NewEstimate

def epsilon_greedy(value, e, seed = None):
    '''
    Implement Epsilon-Greedy policy.
    
    Inputs:
    value: numpy ndarray
            A vector of values of actions to choose from
    e: float
            Epsilon
    seed: None or int
            Assign an integer value to remove the randomness
    
    Outputs:
    action: int
            Index of the chosen action
    '''
    
    assert len(value.shape) == 1
    assert 0 <= e <= 1
    
    if seed != None:
        np.random.seed(seed)
    
    ############################
    # YOUR CODE STARTS HERE
    # short_cut   
    n             = value.shape[0]
    max_index     = value.argmax()
    if max_index == value.argmin():
        probability   = [1/n]*n
    else:
        probability = [e/n]*n
        probability[max_index] = probability[max_index] + (1-e)
        
    assert(sum(probability)==1)
#     print(probability)
    action = np.random.choice(np.arange(n),p=probability)
    # YOUR CODE ENDS HERE
    ############################
    return action

def action_evaluation(env, gamma, v):
    '''
    Convert V value to Q value with model.
    
    Inputs:
    env: OpenAI Gym environment
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
            Discount value
    v: numpy ndarray
            Values of states
            
    Outputs:
    q: numpy ndarray
            Q values of all state-action pairs
    '''
    
    nS = env.nS
    nA = env.nA
    q = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            ############################
            # YOUR CODE STARTS HERE
            q[s][a] = sum([p*(r+ gamma*v[s]) for p,s,r,t in env.P[s][a]])
            # YOUR CODE ENDS HERE
            ############################
    return q

def action_selection(q):
    '''
    Select action from the Q value
    
    Inputs:
    q: numpy ndarray
    
    Outputs:
    actions: int
            The chosen action of each state
    '''
    
    actions = np.argmax(q, axis = 1)    
    return actions 

def render(env, policy):
    '''
    Play games with the given policy
    
    Inputs:
    env: OpenAI Gym environment 
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
    policy: numpy ndarray
            Maps state to action
    '''
    
    state = env.reset()
    terminal = False
    
    while not terminal:
        action = policy[state]
        state, reward, terminal, prob = env.step(action)
        env.render()
        sleep(1)
    
    print('Episode ends. Reward =', reward)
    
def human_play(env):
    '''
    Play the game.
    
    Inputs:
    env: OpenAI Gym environment
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
    '''
    
    print('Action indices: LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3')
    state = env.reset()
    env.render()
    terminal = False
    
    while not terminal:
        action = int(input('Give the environment your action index:'))
        state, reward, terminal, prob = env.step(action)
        env.render()
