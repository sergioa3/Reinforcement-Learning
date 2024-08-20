import numpy as np


#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
class distribution:
    def __init__(self, distribution_dict:dict) -> None:
        d_sum = np.array(list(distribution_dict.values())).sum()
        if d_sum != 1:
            raise Exception('Distribution must sum 1, got {}'.format(d_sum))
        self.dictio = distribution_dict

    def get_state(self):
        values = np.array(list(self.dictio.keys()))
        probabilities = list(self.dictio.values())

        # Generate a sample
        sample = np.random.choice(values, p=probabilities)
        return sample
    
    def get_possible_states(self) -> dict.keys:
        return self.dictio.keys()
    

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

class kernel:
    def __init__(self, state_action_distributions:dict) -> None:
        self.state_action_distributions = state_action_distributions
        #state_action_distributions must be a dictionary with tuple keys (state, action) and values of class distribution
    
    def get_state(self, state, action):
        try:
            sample = self.state_action_distributions[(state, action)].get_state()
        except Exception as e:
            raise e
            
        return sample
    
    def get_distribution(self, state, action)-> distribution:
        try:
            dist = self.state_action_distributions[(state, action)]
        except Exception as e:
            raise e
        return dist
    
    def get_state_actions(self) -> dict.keys:
        return self.state_action_distributions.keys()
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

class rewards:
    def __init__(self, reward_funtion:dict) -> None:
        self.reward_function = reward_funtion
        #reward funtion must be a dictionary with tuple keys (state, action) and values of class float

    def get_reward(self, state, action):
        try:
            reward = self.reward_function[(state, action)]
        except Exception as e:
            raise e
            
        return reward
    
    def get_state_actions(self) -> dict.keys:
        return self.reward_function.keys()

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

#episodic environments will be represented by environments which have absorving states.
#what is an absorving state?
# A state s whose distribution for any action a looks like this: {s:1}.

#that means we are assuming that absorving states are terminal states


#the environment will check for absorving states from the begining and tag them

class environment:
    def __init__(self, kernel:kernel , rewards:rewards, d0:distribution) -> None:
        
        state_actions_k = kernel.get_state_actions()
        state_actions_r = rewards.get_state_actions()

        
        #the set of states must be given by all the possible entries of state in the kernel and the rewards
        #and the consistency of these must be checked:

        # Extract states and actions from state_actions_k and state_actions_r
        states_k = {sa[0] for sa in state_actions_k}
        actions_k = {sa[1] for sa in state_actions_k}

        states_r = {sa[0] for sa in state_actions_r}
        actions_r = {sa[1] for sa in state_actions_r}


        # Compare the sets of states and actions
        same_states = states_r == states_k
        same_actions = actions_r == actions_k

        if not (same_states):
            raise Exception('States from kernel are not consistent with states of reward function.')
        
        if not (same_actions):
            raise Exception('Actions from kernel are not consistent with actions of reward function.')
        
        if not (set(state_actions_r)==set(state_actions_k)):
            raise Exception('State-action pairs from kernel are not consistent with state-action pairs of reward function.')
        

        # New dictionary to store available actions for each state
        state_actions = {}

        # Populate the new dictionary
        for (state, action) in state_actions_r:
            if state not in state_actions:
                state_actions[state] = set()  # Initialize with an empty set
            state_actions[state].add(action)

        self.state_actions = state_actions
        del states_r
        del actions_r
        del states_k
        del actions_k
        del state_actions_r
        del state_actions_k
        
        
        # Check if d0 states is a subset of states
        if not set(d0.get_possible_states()).issubset(set(self.get_all_states())):
            raise Exception('d0 must be a subset of the possible states.')
        
        self.d0 = d0
        self.kernel = kernel
        self.rewards = rewards


        #tag absorving/terminal states:
        terminal_states = set()
        for s in self.get_all_states():
            all_possible_next_states = set()
            actions = self.get_actions(s)
            for a in actions:
                all_possible_next_states.update(set(self.kernel.get_distribution(s,a).get_possible_states()))
            
            if all_possible_next_states == set([s]):
                #state is absorving/terminal
                terminal_states.add(s)

        self.terminal_states = terminal_states

        
    def get_all_states(self)-> dict.keys:
        return self.state_actions.keys()
    
    def get_actions(self, state)-> set:
        #Actions for state
        return self.state_actions[state]
    
    def get_all_actions(self)-> set:
        # Union of all action sets across all states
        all_actions = set.union(*self.state_actions.values())
        return all_actions

    def get_initial_state(self):
        state = self.d0.get_state()

        return state

    def take_action(self, state, action)-> tuple:
        #RETURN WILL BE A TRIPLE
        #(state, reward, boolean)
        #boolean indicates if the state inputed was terminal and therefore environment must me restarted by agent
        terminal = False

        if state in self.terminal_states:
            terminal = True
        try:
            s_next = self.kernel.get_state(state, action)
            r = self.rewards.get_reward(state, action)
        except KeyError:
            raise KeyError('Action "{}" not recognized for state "{}" by environment.'.format(action,state))
        
        return (s_next, r, terminal)

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

class Q_Estimate:
    def __init__(self, environment, action0=None) -> None:
        #structure must be a mapping from (state, action) to values
        state_actions = environment.state_actions
        estimate = {}
        for state in set(state_actions.keys()):
            estimate[state] = {}
            for action in state_actions[state]:
                sa = 0
                if action0:
                    if action in action0:
                        sa = 0.01
                estimate[state][action] = sa
        
        self.estimate = estimate

    def update(self, state, action, sa)-> None:
        if not state in set(self.estimate.keys()):
            raise KeyError('State "{}" not recognized by Q.'.format(state))
        
        if not action in set(self.estimate[state].keys()):
            raise KeyError('Action "{}" not recognized for state "{}" by Q'.format(action, state))
        
        self.estimate[state][action] = sa

        return None
    
    def get_states(self):
        return set(self.estimate.keys())
    
    def argmax_a(self, state)-> tuple:
        #return tuple of (action, value) of maximum Q_estimate value for state
        if not state in set(self.estimate.keys()):
            raise Exception('State "{}" not recognized by Q.'.format(state))

        actions = set(self.estimate[state].keys())

        argmax = np.random.choice(np.array(list(actions)))
        max = self.get(state, argmax)
        for a in actions:
            sa = self.get(state, a)
            if sa > max:
                argmax = a
                max = sa
        return (argmax, max)


    def get(self, state, action)-> float:
        try:
            action_values = self.estimate[state]
        except KeyError:
            raise KeyError('State "{}" not recognized by Q.'.format(state))
        
        try:
            sa = action_values[action]
        except KeyError:
            raise KeyError('Action "{}" not recognized for state "{}" by Q'.format(action, state))
        
        return sa

        
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

class policy:
    def __init__(self, environment) -> None:
        #structure must be a mapping from states to actions
        state_actions = environment.state_actions
        pi = {}
        for state in set(state_actions.keys()):
            actions = state_actions[state]
            #choose random action
            any_action = np.random.choice(np.array(list(actions)))
            pi[state] = any_action
        
        self.pi = pi

    def __init__(self, environment:environment, Q:Q_Estimate, epsilon:float = 0) -> None:
        #structure must be a mapping from states to actions
        state_actions = environment.state_actions
        pi = {}

        if not (0<=epsilon and epsilon<=1):
            raise Exception('Epsilon must be between zero and 1, received {}'.format(epsilon))
        
        for state in set(state_actions.keys()):
            actions = state_actions[state]
            #with probability epsilon choose a random action
            if np.random.binomial(n=1, p=1-epsilon)==1:
                #choose Q greedy action
                action = Q.argmax_a(state)[0]
            else:
                #choose random action
                action = np.random.choice(np.array(list(actions)))
            pi[state] = action
        
        self.pi = pi
    
    def update(self, state, action):
        #this will check if the state is recognized by policy
        if not state in set(self.pi.keys()):
            raise Exception('State "{}" not recognized by policy.'.format(state))
        
        #however the policy will not check wether the action is recognized by environment
        
        self.pi[state] = action

    def get_action(self, state):
        #this will check if the state is recognized by policy
        try:
            action = self.pi[state] 
        except KeyError:
            raise KeyError('State "{}" not recognized by policy.'.format(state))
        return action
        

#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------
  
class agent:
    def __init__(self, environment: environment, discount:float, history_callback: callable = None, action0 = None) -> None:
        
        if not (0<=discount and discount<=1):
            raise Exception('Discount factor must be between zero and 1, received {}'.format(discount))
        
        self.environment = environment
        self.cummulative_reward = 0
        self.discount = discount

        self.state = self.environment.get_initial_state()
        if action0:
            self.Q = Q_Estimate(environment, action0)
        else:
            self.Q = Q_Estimate(environment)
        self.pi = policy(environment, self.Q, epsilon=0)

#        print(self.Q.estimate)
        print(self.pi.pi)

        self.running = False
        self.history = [self.state]

        self.history_callback = history_callback
        

    def restart(self):
        self.cummulative_reward = 0
        self.state = self.environment.get_initial_state()
        self.history = [self.state]
        
    def update(self, state, r):
        self.state = state
        self.history.append(state)
        self.cummulative_reward = self.cummulative_reward + r
        if self.history_callback:
            self.history_callback(self.history, self.cummulative_reward)

    

    def sarsa(self, alpha, epsilon):
        self.running = True
        while True:
            print(self.cummulative_reward)
            self.restart()
            terminated = False
            s = self.state
            
            if np.random.binomial(n=1, p=1-epsilon)==1:
                #choose Q greedy action
                a = self.pi.get_action(s)
            else:
                #choose random action
                actions = self.environment.get_actions(s)
                a = np.random.choice(np.array(list(actions)))

            while not terminated:
                
                s_next, r, terminated = self.environment.take_action(s, a)
                #print(s, a, s_next)



                if np.random.binomial(n=1, p=1-epsilon)==1:
                    #choose Q greedy action
                    a_next = self.pi.get_action(s_next)
                else:
                    #choose random action
                    actions = self.environment.get_actions(s_next)
                    a_next = np.random.choice(np.array(list(actions)))

                q = self.Q.get(s, a)
                q_next = self.Q.get(s_next, a_next)
                new_q = q + alpha * ( r + self.discount*q_next - q )

                self.Q.update(s,a, new_q)
                #if new_q < q:
                self.pi.update(s, self.Q.argmax_a(s)[0])

                s = s_next
                a = a_next
                self.update(s, r)
                
            

