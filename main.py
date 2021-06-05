from env import World
import numpy as np


def run_AB_code(num_episodes):
   for n_episode in np.arange(0, num_episodes):
       env.reset()
       done = False
       t = 0
       env.show()
       while not done:
           env.render()
           action = np.random.randint(1, env.nActions ) # take a random action
           # print("Action taken:%s" %(states[action])) #print the actiokn - need to delete
           next_state, reward, done = env.step(action)  # observe next_state and reward
           # print("Next_state:%s" %(next_state))
           env.close()
           t += 1
           if done:
               print("Episode",n_episode + 1, "finished after {} timesteps".format(t + 1))
               break



def bulid_transision_N():
    '''
    this function generates PSS' for North action

    Returns
    -------
    trans_N : transision model for all states action = North.  
    np matrix size nStates X nStates 


    '''
    left_prob = 0.1
    right_prob = 0.1
    next_prob = 0.8
    trans_N =  np.zeros((env.nStates,env.nStates))
    for state in range(env.nStates):
        if state + 1 not in env.stateTerminals:
            (I,J) = np.unravel_index(state,shape=(env.nRows,env.nCols), order='F')
            
            if I!=0:
                trans_N[state][state-1] += next_prob
            else: trans_N[state][state] += next_prob
            
            if J!=0:        
                trans_N[state][state-env.nRows] += left_prob
            else:  trans_N[state][state] += left_prob
             
            if J!= (env.nCols-1):
                trans_N[state][state+env.nRows] += right_prob
            else: trans_N[state][state] += right_prob   
            
    
    return trans_N




def bulid_transision_E():
    '''
    this function generates PSS' for East action

    Returns
    -------
    trans_E : transision model for all states action = East.  
    np matrix size nStates X nStates 

    '''
    left_prob = 0.1
    right_prob = 0.1
    next_prob = 0.8
    trans_E =  np.zeros((env.nStates,env.nStates))
    for state in range(env.nStates):
        if state + 1 not in env.stateTerminals:
            (I,J) = np.unravel_index(state,shape=(env.nRows,env.nCols), order='F')
            
            if I!=0:
                trans_E[state][state-1] += left_prob 
            else:  trans_E[state][state] += left_prob

            if I!= (env.nRows-1):
                trans_E[state][state+1] += right_prob
            else:  trans_E[state][state] += right_prob
            
            if J!= (env.nCols - 1):
                trans_E[state][state+env.nRows] += next_prob
            else:  trans_E[state][state] += next_prob

            
    return trans_E  
    
    
    
def bulid_transision_S():
    '''
    this function generates PSS' for South action

    Returns
    -------
    trans_S : transision model for all states action = South.  
    np matrix size nStates X nStates 

    '''
    left_prob = 0.1
    right_prob = 0.1
    next_prob = 0.8
    trans_S =  np.zeros((env.nStates,env.nStates))
    for state in range(env.nStates):
        if state + 1 not in env.stateTerminals:
            (I,J) = np.unravel_index(state,shape=(env.nRows,env.nCols), order='F')
            
            if J!=0:
                trans_S[state][state-env.nRows] += right_prob
            else: trans_S[state][state] += right_prob
                
            if J!= (env.nCols-1):
                trans_S[state][state+env.nRows] += left_prob
            else: trans_S[state][state] += left_prob

            if I!= (env.nRows-1):
                trans_S[state][state+1] += next_prob
            else: trans_S[state][state] += next_prob

        
    return trans_S
    
    
    
def bulid_transision_W():
    '''
    this function generates PSS' for West action

    Returns
    -------
    trans_W : transision model for all states action = West. 
    np matrix size nStates X nStates 


    '''
    left_prob = 0.1
    right_prob = 0.1
    next_prob = 0.8
    trans_W =  np.zeros((env.nStates,env.nStates))
    for state in range(env.nStates):
        if state + 1 not in env.stateTerminals:
            (I,J) = np.unravel_index(state,shape=(env.nRows,env.nCols), order='F')
            
            if I!=0:
                trans_W[state][state-1] += right_prob 
            else: trans_W[state][state] += right_prob
            
            if I!= (env.nRows-1):
                trans_W[state][state+1] += left_prob
            else: trans_W[state][state] += left_prob
           
            if J!=0:
                trans_W[state][state-env.nRows] += next_prob
            else:  trans_W[state][state] += next_prob


    return trans_W 
    

def build_transision_model():
    '''
    this function generates PSS' for all actions

    Returns
    -------
    transision_MDP : transision model for all states for all actions.
    3D np matrix size nActions X nStates X nStates 
    '''
    transision_MDP = np.zeros((env.nActions,env.nStates,env.nStates))
    # actions = ['N', 'E', 'S', 'W']
    transision_MDP[0] = bulid_transision_N()
    transision_MDP[1] = bulid_transision_E()
    transision_MDP[2] = bulid_transision_S()
    transision_MDP[3] = bulid_transision_W()
      

    return transision_MDP



def bulid_reward_function():
    '''
    this function generates reward model for all states

    Returns
    -------
    rewards : reward per state.  
    vector size nStates X 1
    '''
    
    
    rewards = np.zeros(env.nStates)
    for state in range(env.nStates):
        if state + 1 in env.stateHoles:
            rewards[state] = -1 
        if state + 1 in env.stateGoal:
            rewards[state] = 1 
        
    
    return rewards



def epsilon_decay(epsilon, decay_method ,n):
   
    '''
    this function generates epsilon for epsilon decay method used for e-greedy
    
    
    :params: epsilon: value of previous epsilon 
             type: float
             decay_method: decay_method to use for e-greedy
             type: string
            
    :return: epsilon: new epsilon
             type: float
    
    '''
    
    if decay_method == "1/((n/10)+1)" :
        epsilon = 1/((n/10)+1)
        
        
    if decay_method == "0.9*e" :
        epsilon = 0.9 * epsilon
            
    return epsilon
    


def run_dynamic_programming(gamma, teta):
    '''
    this function preforms dynamic programing algorithem to find optimal policy.
    
    :params: gamma: value of gamma for discounted return
             type: float
             teta: value of teta
             type: float
             
    :returns: state_values: optimal state values matrix
             type: numpy vector size nStates X 1
             best_policy: optimal policy matrix , size nStates X nActions, 1 indicates optimal action
             type: numpy matrix size nStates X nActions
             

    '''
    transision_MDP = build_transision_model()
    reward_function = bulid_reward_function()
    state_values = env.value_iteration(transision_MDP, reward_function, gamma, teta)
    best_policy = env.find_policy_DP(state_values, transision_MDP, reward_function, gamma) 
    env.plot_value(state_values)
    env.plot_policy(best_policy)
    
    return state_values, best_policy

def run_MC_control(num_episodes, alpha, epsilon, gamma, decay_method,show):
    
    '''
    this function preforms run_MC_control algorithem to find optimal policy.
    
    :params: num_episodes: number of episodes to run 
             type: int
             alpha: learning rate
             type: float
             epsilon: initial epsilon for e-greedy method
             type: float
             gamma: value of gamma for discounted return
             type: float
             decay_method: decay_method to use for e-greedy
             type: string
             show: indicator that sets if to show agent progress in gridworld or not
             type: boolean
             
    :returns: q_values: optimal action_state_values matrix
             type: numpy matrix size nStates X nActions
             best_policy: optimal policy matrix , size nStates X nActions, 1 indicates optimal action
             type: numpy matrix size nStates X nActions
             

    '''
    
    
    print("runing MC")
    q_values = np.zeros([env.nStates,env.nActions])
    for n_episode in np.arange(1, num_episodes+1):
       env.reset()
       q_values = env.mc_control(alpha, epsilon, gamma, q_values ,n_episode, show)
       epsilon = epsilon_decay(epsilon, decay_method ,n_episode)
    
    best_policy = env.find_policy(q_values)
    state_values = env.get_state_values(q_values)
 #   if show == True:
    env.plot_policy(best_policy)
    env.plot_qvalue(q_values)
    env.plot_value(state_values)
    
    return state_values, q_values, best_policy


def run_SARSA(num_episodes, alpha, epsilon, gamma, decay_method,show):
    
    '''
    this function preforms SARSA algorithem to find optimal policy.
    
    :params: num_episodes: number of episodes to run 
             type: int
             alpha: learning rate
             type: float
             epsilon: initial epsilon for e-greedy method
             type: float
             gamma: value of gamma for discounted return
             type: float
             decay_method: decay_method to use for e-greedy
             type: string
             show: indicator that sets if to show agent progress in gridworld or not
             type: boolean
             
    :returns: q_values: optimal action_state_values matrix
             type: numpy matrix size nStates X nActions
             best_policy: optimal policy matrix , size nStates X nActions, 1 indicates optimal action
             type: numpy matrix size nStates X nActions
             

    '''   
    
    print("runing SARSA")
    q_values = np.zeros([env.nStates,env.nActions])
    for n_episode in np.arange(1, num_episodes+1):
       env.reset()
       q_values = env.sarsa(alpha, epsilon, gamma, q_values ,n_episode, show)
       epsilon = epsilon_decay(epsilon, decay_method ,n_episode)
    
    best_policy = env.find_policy(q_values)
    state_values = env.get_state_values(q_values)
  #  if show == True:
    env.plot_policy(best_policy)
    env.plot_qvalue(q_values)
    env.plot_value(state_values)
    
    return state_values,q_values, best_policy
    
def run_Qlearning(num_episodes, alpha, epsilon, gamma, decay_method,show):
    
    '''
    this function preforms Qlearning algorithem to find optimal policy.
    
    :params: num_episodes: number of episodes to run 
             type: int
             alpha: learning rate
             type: float
             epsilon: initial epsilon for e-greedy method
             type: float
             gamma: value of gamma for discounted return
             type: float
             decay_method: decay_method to use for e-greedy
             type: string
             show: indicator that sets if to show agent progress in gridworld or not
             type: boolean
             
    :returns: q_values: optimal action_state_values matrix
             type: numpy matrix size nStates X nActions
             best_policy: optimal policy matrix , size nStates X nActions, 1 indicates optimal action
             type: numpy matrix size nStates X nActions
             

    '''
    print("runing Q-learning")
    q_values = np.zeros([env.nStates,env.nActions])
    for n_episode in np.arange(1, num_episodes+1):
       env.reset()
       q_values = env.q_learning(alpha, epsilon, gamma, q_values ,n_episode, show)
       epsilon = epsilon_decay(epsilon, decay_method ,n_episode)
    
    best_policy = env.find_policy(q_values)
    state_values = env.get_state_values(q_values)
 #   if show == True:
    env.plot_policy(best_policy)
    env.plot_qvalue(q_values)
    env.plot_value(state_values)
    
    return state_values, q_values, best_policy



def calc_mse(state_values_DP,state_values_alg):
    '''
    this function calculates MSE in oreder to choose best parmas.
    :params: state_values_DP: optimal state values recived from runing DP
            type: np vector size nStates X 1
            state_values_alg: optimal state values approximation recived from runing MC/sarsa/q-learning
            type: np vector size nStates X 1
            
    rerurn: MSE
            type: float
    '''
    
    mse_nom = 0
    for state in range(env.nStates):
        mse_nom = mse_nom + pow((state_values_DP[state-1]-state_values_alg[state-1]),2)
    mse = mse_nom/env.nStates
    return mse


def get_best_params(dict_measures):
    '''
    this function returns the best params for model based on MSE values
    :params: dict_measures: dictonary contaning MSE score per params combo.  {key = alpha,decay,epsilon ,value = MSE}
            type: dictonary {string:float}
        
            
    rerurn: best_combo: key for optimal value = min MSE
            type: string
    '''
    best_combo = min(dict_measures, key=dict_measures.get)
    return best_combo
        

def parameters_tuninnig(state_values_DP):
    
    '''
    this function preforms tuning on 500 episodes of tested parameters
    
    :params: state_values_DP: ptimal state values recived from runing DP
            type: np vector size nStates X 1
    
    :return: dict_algs: dictonry contanins best parameters found for MC/sarsa/q-learning {key =algorithem_name ,value = alpha,decay,epsilon}
            type: dictonary {string:string}
    
    
    '''
    
    
    dict_algs ={}
    algs = ["MC","SARSA","Qlearning"]
    episodes = 500
    dict_measures = {}
    alphas = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    decays = ["1/((n/10)+1)", "0.9*e"] 
#    epsilons = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    gamma = 0.9
    epsilon = 0.9
   
    for alg in algs:
        for alpha in alphas:
            for decay in decays:
       #         for epsilon in epsilons:
                    dict_string = "alpha: " +str(alpha) + " decay:" + str(decay) +" epsilon:" + str(epsilon)
                    if alg == "MC":
                       state_values_alg,q_values_MC_control ,best_policy_MC_control = run_MC_control(episodes, alpha, epsilon, gamma, decay,show=False)
                    if alg == "SARSA":
                      state_values_alg,q_values_sarsa ,best_policy_sarsa = run_SARSA (episodes, alpha, epsilon, gamma, decay,show=False)
                    if alg == "Qlearning":
                       state_values_alg,q_values_Qlearning ,best_policy_Qlearning = run_Qlearning (episodes, alpha, epsilon, gamma, decay ,show=False)
                       
                    mse = calc_mse(state_values_DP,state_values_alg)
                    dict_measures[dict_string] = mse
                    print(dict_string)
                    print(mse)
        
        best_params = get_best_params(dict_measures)
        dict_algs[alg] = best_params                      
      
    return dict_algs



if __name__ == "__main__":

    env = World()
#   run_AB_code(num_episodes = 10)


    state_values_DP, best_policy_DP = run_dynamic_programming(0.9 , 0.0001)
  #  best_params = parameters_tuninnig = parameters_tuninnig(state_values_DP)
    state_values_MC_control, q_values_MC_control ,best_policy_MC_control = run_MC_control(5000, 0.01 , 0.9, 0.9, decay_method = "1/((n/10)+1)",show=False)
    state_values_sarsa, q_values_sarsa ,best_policy_sarsa = run_SARSA (5000, 0.1 , 0.9, 0.9, decay_method = "1/((n/10)+1)",show=False)
    state_values_Qlearning, q_values_Qlearning ,best_policy_Qlearning = run_Qlearning (5000, 0.3 , 0.9, 0.9, decay_method = "1/((n/10)+1)",show=False)
    
    
    
    
































