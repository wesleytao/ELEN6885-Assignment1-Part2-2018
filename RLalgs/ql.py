import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
    """

    Q = np.zeros((env.nS, env.nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    for this_episode in range(num_episodes):
        init_state = np.random.choice(range(env.nS)) # something
        current_state = init_state
        terminal = False
#         print("--------------------episode {}---------".format(this_episode))
        
        while(not terminal): # not terminal state
            # choose the action based on epsilon greedy
            # update Q value based on the best next state Q value
#             print("current state {}".format(current_state))
            action = epsilon_greedy(Q[current_state], e)
#             print("action {}".format(action) )
            info_array = np.array(env.P[current_state][action])
            n_next_state = info_array.shape[0]
            rand_indx = np.random.choice(range(n_next_state), p = info_array[:,0]) # realized next state
            
            next_state = int(info_array[rand_indx,1]) # next state
            reward     = info_array[rand_indx,2] # reward
            terminal = info_array[rand_indx,3]
            
            Q[current_state][action] = Q[current_state][action] + lr * (reward +gamma * Q[next_state].max() - Q[current_state][action])
            # 
            current_state = next_state


    # YOUR CODE ENDS HERE
    ############################

    return Q