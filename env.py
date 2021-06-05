import numpy as np
import matplotlib.pyplot as plt
import _env

class World(_env.Hidden):

    def __init__(self):

        self.nRows = 4
        self.nCols = 5
        self.stateInitial = [4]
        self.stateTerminals = [1, 2,  10, 12, 17, 20]
        self.stateObstacles = []
        self.stateHoles = [1, 2,  10, 12, 20]
        self.stateGoal = [17]
        self.nStates = 20
        self.nActions = 4

        self.observation = [4]  # initial state

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def _plot_world(self):

        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        stateGoal      = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateObstacles:
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.3")
            plt.plot(xs, ys, "black")
        for i in stateTerminals:
            #print("stateTerminal", i)
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            #print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            #print("coord", xs,ys)
            plt.fill(xs, ys, "0.6")
            plt.plot(xs, ys, "black")
        for i in stateGoal:
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            #print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            #print("coord", xs,ys)
            plt.fill(xs, ys, "0.9")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')



    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center', verticalalignment='center')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_value(self, valueFunction):

        """
        plot state value function V

        :param policy: vector of values of size nStates x 1
        :return: None
        """

        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in self.stateHoles + stateObstacles + self.stateGoal: 
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(np.round(valueFunction[k],4),3)), fontsize=16, horizontalalignment='center', verticalalignment='center')
                k += 1
        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right',verticalalignment='bottom')
                k += 1

        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_policy(self, policy):

        """
        plot (stochastic) policy

        :param policy: matrix of policy of size nStates x nActions
        :return: None
        """
        # remove values below 1e-6
        policy = policy * (np.abs(policy) > 1e-6)


        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        #policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        # generate mesh for grid world
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        # generate locations for policy vectors
        #print("X = ", X)
        X1 = X.transpose()
        X1 = X1[:-1, :-1]
        #print("X1 = ", X1)
        Y1 = Y.transpose()
        Y1 = Y1[:-1, :-1]
        #print("Y1 =", Y1)
        X2 = X1.reshape(-1, 1) + 0.5
        #print("X2 = ", X2)
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        #print("Y2 = ", Y2)
        # reshape to matrix
        X2 = np.kron(np.ones((1, nActions)), X2)
        #print("X2 after kron = ", X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        #print("X2 = ",X2)
        #print("Y2 = ",Y2)
        # define an auxiliary matrix out of [1,2,3,4]
        mat = np.cumsum(np.ones((nStates , nActions)), axis=1).astype("int64")
        #print("mat = ", mat)
        # if policy vector (policy deterministic) turn it into a matrix (stochastic policy)
        #print("policy.shape[1] =", policy.shape[1])
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
            policy = policy.astype("int64")
            print("policy inside", policy)
        # no policy entries for obstacle and terminal states
        index_no_policy = stateObstacles + stateTerminals
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        #print("index_policy", index_policy)
        #print("index_policy[0]", index_policy[0:2])
        mask = (policy > 0) * mat
        #print("mask", mask)
        #mask = mask.reshape(nRows, nCols, nCols)
        #X3 = X2.reshape(nRows, nCols, nActions)
        #Y3 = Y2.reshape(nRows, nCols, nActions)
        #print("X3 = ", X3)
        # print arrows for policy
        # [N, E, S, W] = [up, right, down, left] = [pi, pi/2, 0, -pi/2]
        alpha = np.pi - np.pi / 2.0 * mask
        #print("alpha", alpha)
        #print("mask ", mask)
        #print("mask test ", np.where(mask[0, :] > 0)[0])
        self._plot_world()
        for i in index_policy:
            #print("ii = ", ii)
            ax = plt.gca()
            #j = int(ii / nRows)
            #i = (ii + 1 - j * nRows) % nCols - 1
            #index = np.where(mask[i, j] > 0)[0]
            index = np.where(mask[i, :] > 0)[0]
            #print("index = ", index)
            #print("X2,Y2", X2[ii, index], Y2[ii, index])
            h = ax.quiver(X2[i, index], Y2[i, index], np.cos(alpha[i, index]), np.sin(alpha[i, index]), color='b')
            #h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]),0.3)

        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right', verticalalignment='bottom')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_qvalue(self, Q):
        """
        plot Q-values

        :param Q: matrix of Q-values of size nStates x nActions
        :return: None
        """
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        stateObstacles = self.stateObstacles

        fig = plt.plot(1)

        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateObstacles + stateGoal: 
                    #print("Q = ", Q)
                    plt.text(i + 0.5, j - 0.15, str(self._truncate(Q[k, 0], 3)), fontsize=6,
                             horizontalalignment='center', verticalalignment='top', multialignment='center')
                    plt.text(i + 0.95, j - 0.5, str(self._truncate(Q[k, 1], 3)), fontsize=6, #values were adjusted
                             horizontalalignment='right', verticalalignment='center', multialignment='right')
                    plt.text(i + 0.5, j - 0.85, str(self._truncate(Q[k, 2], 3)), fontsize=6,
                             horizontalalignment='center', verticalalignment='bottom', multialignment='center')
                    plt.text(i + 0.01, j - 0.5, str(self._truncate(Q[k, 3], 3)), fontsize=6, #values were adjusted
                             horizontalalignment='left', verticalalignment='center', multialignment='left')
                    # plot cross
                    plt.plot([i, i + 1], [j - 1, j], 'black', lw=0.5)
                    plt.plot([i + 1, i], [j - 1, j], 'black', lw=0.5)
                k += 1

        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateTerminals(self):

        return self.stateTerminals

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateObstacles(self):

        return self.stateObstacles

    def get_stateGoal(self):

        return self.stateGoal


    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions


    def step(self,action):

        nStates = self.nStates
        stateGoal = self.get_stateGoal()
        stateTerminals = self.get_stateTerminals()

        state = self.observation[0]


        # generate reward and transition model
        p_success = 0.8
        r = -0.04
        self.get_transition_model(p_success)
        self.get_reward_model(r,p_success)
        Pr = self.transition_model
        R = self.reward

        prob = np.array(Pr[state-1, :, action])
        #print("prob =", prob)
        next_state = np.random.choice(np.arange(1, nStates + 1), p = prob)
        #print("state = ", state)
        #print("next_state inside = ", next_state)
        #print("action = ", action)
        reward = R[state-1, next_state-1, action]
        #print("reward = ", R[:, :, 0])
        observation = next_state

        #if (next_state in stateTerminals) or (self.nsteps >= self.max_episode_steps):
        if (next_state in stateTerminals):
            done = True
        else:
            done = False

        self.observation = [next_state]


        return observation, reward, done


    def reset(self, *args):


        nStates = self.nStates

        if not args:
            observation = self.stateInitial
        else:
            observation = []
            while not (observation):
                observation = np.setdiff1d(np.random.choice(np.arange(1, nStates +  1, dtype = int)), self.stateHoles + self.stateObstacles + self.stateGoal)
        self.observation = observation



    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation #observation
        state = observation[0]


        J, I = np.unravel_index(state - 1, (nRows, nCols), order='F')



        J = (nRows -1) - J



        circle = plt.Circle((I+0.5,J+0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()


    def close(self):
        plt.pause(0.3) #0.5
        plt.close()

    def show(self):
        plt.ion()
        plt.show()
        
    
    def choose_action_e_greedy(self, state, epsilon, q_values):
        
          
        '''
        this function chooses next action using e_greedy algorithem
    
        :params: state: current state
                 type: int
                 epsilon: epsilon for e-greedy method
                 type: float
                 q_values: action_state_values matrix
                 type: numpy matrix size nStates X nActions

             
       :return: action_to_take: next action to take
                type: int 0 = north, 1 = east , 2 =south , 3 = west
             
        '''
        
        
        pi = epsilon/self.nActions
        action_state_value = q_values[state-1,:]
        p_action = ['NA']*self.nActions
        optimal_action = np.argmax(action_state_value)
        for action in range(0,self.nActions):
            if action == optimal_action:
                p_action[action] = pi + 1 - epsilon
            else:
                p_action[action] = pi
        action_to_take = np.random.choice(np.arange(0,self.nActions),p=p_action)
        return action_to_take
   
    
    def genrate_episode_e_greedy(self, state, epsilon, q_values, n_episode, show):
        
        '''
        this function generates full episode to use for MC algorithem
    
        :params: state: current state
                 type: int
                 epsilon: epsilon for e-greedy method
                 type: float
                 q_values: action_state_values matrix
                 type: numpy matrix size nStates X nActions
                 n_episode: current episode value
                 type: int
                 show: indicator that sets if to show agent progress in gridworld or not
                 type: boolean


             
       :return: state_action_reward_list: state, action, reward for time t in this episode
                type: list of lists: state_action_reward_tuple [state, action, reward]
             
        '''
        
        
        state_action_reward_list = []
        state_action_reward_tuple = []
        done = False
        action = self.choose_action_e_greedy(state, epsilon, q_values)
        state_action_reward_tuple = [state,action,0]
        state_action_reward_list.append(state_action_reward_tuple)
        t = 0
        while not done:
            if show == True:
                self.render()
            next_state, reward, done = self.step(action)  # observe next_state and reward
            next_action = self.choose_action_e_greedy(next_state, epsilon, q_values)
            state_action_reward_tuple = [next_state, next_action, reward]
            state_action_reward_list.append(state_action_reward_tuple)
            action = next_action
            state = next_state
            t += 1
            if done:
            #    print("Episode",n_episode , "finished after {} timesteps".format(t + 1))
                break
            
        return state_action_reward_list
    
    def get_q_greedy(self, state, q_values):
       
        '''
        this function return the optimal q for q-greedy algorithem used in q-learning algorithem
    
        :params: state: current state
                 type: int
                 q_values: action_state_values matrix
                 type: numpy matrix size nStates X nActions
                 
             
       :return: optimal_q: 
                type: float

        '''
        action_state_value = q_values[state-1,:]
        optimal_q = np.max(action_state_value)
        return optimal_q
    
   
    def update_q(self, q, alpha, next_q, reward, gamma):
        
        '''
        this function calculates q per state
    
        :params: q: current q value for state 
                 type: float
                 alpha: learning rate
                 type: float
                 next_q: q value for next state
                 type: float
                 reward: reward of next state
                 type: float
                 gamma: gamma value for discounted return
                 type: float
                 
             
       :return: updated_q: updated q value for current state 
                type: float

        '''
        
        
        updated_q = q + alpha*(reward + gamma*(next_q) - q)
        return updated_q
    
       
   
    def update_q_return(self, q, alpha, G):
        
         
        '''
        this function calculates q based on expected return used in MC algorithem
    
        :params: q: current q value for state 
                 type: float
                 alpha: learning rate
                 type: float
                 G: current return
                 type: float
                 
             
       :return: updated_q: updated q value for current state 
                type: float

        '''
        
        updated_q = q + alpha*(G - q)
        return updated_q



    
    def get_return(self, episode, gamma):
        
        '''
        this function returns the return value for giving episode used for MC algorithem
    
        :params: episode: full list of action, reward for time t in this episode
                 type: list of lists: state_action_reward_tuple [state, action, reward]            
             
       :return: G:  return
                type: float

        '''
        
        
        G = 0
        power = 0
        for tuple_t in episode:
            reward = tuple_t[2]
            G = G + reward * pow(gamma,power)
            
        return G
            
   
    def get_action_values(self,current_state, mdp_transition_model, reward_function, state_values, gamma):
        
        '''
        this function claculates the action_values used in value iteration in DP
    
        :params: current_state: 
                 type: int
                 mdp_transition_model: transision model for all states for all actions.
                 type: 3D np matrix size nActions X nStates X nStates 
                 reward_function:  reward per state.  
                 type: vector size nStates X 1
                 state_values:  state values matrix.  
                 type: vector size nStates X 1
                 gamma: gamma value for return calaculation
                 type: float
             
       :return: action_vec:  return
                type: np vector size nActios X 1

        '''
        
        
        action_reward = -0.04
        action_vec = np.zeros(self.nActions)
        for action in range(self.nActions):
            mdp_action = mdp_transition_model[action,:,:]
            reward = 0
            E_state_value = 0
            v = 0
            for next_state in range(1,self.nStates+1):
                reward = reward + (action_reward + reward_function[next_state-1]) * mdp_action[current_state-1,next_state-1] # reward * PSS'
                E_state_value = E_state_value + gamma * (state_values[next_state-1] * mdp_action[current_state-1,next_state-1]) #gamma * V(S) * PSS'
                v = reward + E_state_value
          
            action_vec[action] =   v

        return action_vec
  


    def update_v(self,current_state, mdp_transition_model, reward_function, state_values, gamma):
        
        '''
        this function updates v for DP value iteration
    
        :params:current_state:
                type: int
                 mdp_transition_model: transision model for all states for all actions.
                 type: 3D np matrix size nActions X nStates X nStates 
                 reward_function:  reward per state.  
                 type: vector size nStates X 1
                 state_values:  state values matrix.  
                 type: vector size nStates X 1 
                 gamma: gamma value for return calaculation
                 type: float
             
        :return: v: updated v value
                type: float

        '''
        
        action_vec = self.get_action_values(current_state, mdp_transition_model, reward_function, state_values, gamma)
        v = max(action_vec)
        return v
      
    
   
    def value_iteration(self, mdp_transition_model, reward_function, gamma, teta):
        
        '''
        this function preforms value_iteration for DP 
    
        :params:
                 mdp_transition_model: transision model for all states for all actions.
                 type: 3D np matrix size nActions X nStates X nStates 
                 reward_function:  reward per state.  
                 type: vector size nStates X 1
                 state_values:  state values matrix.  
                 type: vector size nStates X 1 
                 gamma: gamma value for return calaculation
                 type: float
             
        :return: state values: optimal state values matrix 
                type: numpy vector size nStates X 1

        '''
        
        
        state_values = np.zeros(self.nStates)
        delta = teta 
        while delta >= teta:
            delta = 0
            for state in range(1,self.nStates+1):
                if state  not in self.stateTerminals:
                   v = state_values[state-1]
                   updeated_v = self.update_v(state, mdp_transition_model, reward_function, state_values, gamma)
                   delta = max(delta, abs(v-updeated_v)) 
                   state_values[state -1] = updeated_v
                   
        return state_values
     
    
    
   
    def mc_control(self, alpha, epsilon, gamma, q_values, n_episode, show):
        
        '''
        this function preforms full episode of  MC algorithem
    
        :params: alpha: learning rate
                 type: float
                 epsilon: epsilon for e-greedy method
                 type: float
                 gamma: gamma value for return calculation
                 type: float
                 q_values: action values matrix
                 type: numpy matrix size nStates X nActions
                 n_episode: current episode
                 type: int
                 show: indicator that sets if to show agent progress in gridworld or not
                 type: boolean


             
        :return: q_values: action_state_values matrix
                 type: numpy matrix size nStates X nActions
             
        '''
        
        state_counter_vector = np.zeros(self.nStates)
        state = self.observation[0]
        episode = self.genrate_episode_e_greedy(state, epsilon, q_values, n_episode, show)
        index = 0
        for tuple_t in episode[0:-1]: #till T-1
            state , action , reward = tuple_t[0] , tuple_t[1], tuple_t[2]
            q = q_values[state-1][action]
            if state_counter_vector[state-1] == 0 :
                G = self.get_return(episode[index + 1:],gamma)
                q = self.update_q_return(q, alpha, G)
                q_values[state-1][action] = q
                
            state_counter_vector[state-1] = state_counter_vector[state-1] + 1
            index = index + 1
    
        return q_values
    
  
    
    def sarsa(self, alpha, epsilon, gamma, q_values, n_episode, show):
        
        '''
        this function preforms full episode of  SARSA algorithem
    
        :params: alpha: learning rate
                 type: float
                 epsilon: epsilon for e-greedy method
                 type: float
                 gamma: gamma value for return calculation
                 type: float
                 q_values: action values matrix
                 type: numpy matrix size nStates X nActions
                 n_episode: current episode
                 type: int
                 show: indicator that sets if to show agent progress in gridworld or not
                 type: boolean


             
        :return: q_values: action_state_values matrix
                 type: numpy matrix size nStates X nActions
             
        '''
        
        
        state = self.observation[0]
        done = False
        t = 0
        action = self.choose_action_e_greedy(state, epsilon, q_values)
        q = q_values[state-1][action]
        if show == True: 
            self.show()
        while not done:
            if show == True: 
                self.render()
            next_state, reward, done = self.step(action)  # observe next_state and reward
           # print("Next_state:%s" %(next_state))
            next_action = self.choose_action_e_greedy(next_state, epsilon, q_values)
            next_q = q_values[next_state-1][next_action]
            q = self.update_q(q, alpha, next_q, reward, gamma)
            q_values[state-1][action] = q
            t += 1
            action = next_action
            state = next_state
            if done:
            #    print("Episode",n_episode , "finished after {} timesteps".format(t + 1))
                break
    
        return q_values
    
    
    
    def q_learning(self, alpha, epsilon, gamma, q_values, n_episode, show):
        
        '''
        this function preforms full episode of  q_learning algorithem
    
        :params: alpha: learning rate
                 type: float
                 epsilon: epsilon for e-greedy method
                 type: float
                 gamma: gamma value for return calculation
                 type: float
                 q_values: action values matrix
                 type: numpy matrix size nStates X nActions
                 n_episode: current episode
                 type: int
                 show: indicator that sets if to show agent progress in gridworld or not
                 type: boolean


             
        :return: q_values: action_state_values matrix
                 type: numpy matrix size nStates X nActions
             
        '''
        
        state = self.observation[0]
        done = False
        t = 0
        if show == True: 
            self.show()
        while not done:
            if show == True: 
                self.render()
            action = self.choose_action_e_greedy(state, epsilon, q_values)
            q = q_values[state-1][action]
            next_state, reward, done = self.step(action)  # observe next_state and reward
           # print("Next_state:%s" %(next_state))
            next_q_optimal = self.get_q_greedy(next_state,q_values)
            q = self.update_q(q, alpha, next_q_optimal, reward, gamma)
            q_values[state-1][action] = q
            t += 1
            state = next_state
            if done:
            #    print("Episode",n_episode, "finished after {} timesteps".format(t + 1))
                break
    
        return q_values
    
    
    
    
    def find_policy(self, q_values):  
        
        '''
        this function returns optimal policy based on action_state_values
    
        :params: q_values: action_state_values matrix
                 type: numpy matrix size nStates X nActions
            
             
       :return: policy_matrix:  optimal policy matrix , size nStates X nActions, 1 indicates optimal action
                type: numpy matrix size nStates X nActions

        '''
        policy_matrix = np.zeros((self.nStates,self.nActions))
        for state in range(1,self.nStates+1):
            if state not in self.stateTerminals:
                optimal_action = np.argmax(q_values[state-1,:])
                policy_matrix[state-1][optimal_action] = 1
                
        return policy_matrix
    
    
    
    
    
    def find_policy_DP(self, state_values, mdp_transition_model, reward_function, gamma):    
        
        '''
        this function returns the optimal policy for optimal values DP
    
        :params: state_values:  state values matrix.  
                 type: vector size nStates X 1 
                 mdp_transition_model: transision model for all states for all actions.
                 type: 3D np matrix size nActions X nStates X nStates 
                 reward_function:  reward per state.  
                 type: vector size nStates X 1
                 gamma: gamma value for return calaculation
                 type: float
             
       :return: optimal policy matrix , size nStates X nActions, 1 indicates optimal action
                type: numpy matrix size nStates X nActions

        '''
        policy_matrix = np.zeros((self.nStates,self.nActions))
        for state in range(1,self.nStates+1):
            if state not in self.stateTerminals:
               action_values = self.get_action_values(state,mdp_transition_model, reward_function, state_values, gamma)
               optimal_action = np.argmax(action_values)
               policy_matrix[state-1][optimal_action] = 1
                
                
        return policy_matrix
 
    def get_state_values(self, q_values):
        
        
        '''
        this function returns optimal state values matrix based on final q_values
    
        :params: q_values: action_state_values matrix
                 type: numpy matrix size nStates X nActions
            
             
       :return: values_vector:  state values vector
                type: numpy vector size nStates X 1

        '''
        
        
        
        values_vector = np.zeros(self.nStates)
        for state in range(1,self.nStates+1):
            state_value = np.max(q_values[state-1,:])
            values_vector[state-1] = state_value
        
        return values_vector
    