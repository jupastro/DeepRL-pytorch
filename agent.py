
import numpy as np
import torch
import collections

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 3300
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # The discrete action for the model
        self.discrete_action=None
      
        #Initialise both classes as subclasses of agent to use them easily through the code but careful not to restart it over and over
        self.dqn = DQN()
        self.buffer=ReplayBuffer()
        #Variable to know the episode we're in:
        self.episode_num=1
        #Variable to know the length of the minibatch
        self.minibatch_size=128
        #Variables to know the decay rate of epsilon and the initial epsilon
        self.decay_epsilon=0.999
        self.epsilon=1
        #to store the av_loss of the net over each episode to check
        self.av_loss=0
        #set the velocity of the movement
        self.mov_magnitude=0.02
        #set the beggining of the random exploration and the number of episodes
        self.episodes_of_bias_random_exploration=3
        self.random_exploration=True
        #set if there is a prioritize sampling
      
        #set if the agent reaches the goal performing greed policy evaluation
        self.reached_goal=False
        #set a number between 0-1 to make it avoid walls
        self.penalisation_obstacles=0.2
        self.t_training=True
        self.left_environment=True# to decide to explore initially right or left
    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self,check_progress=False):
        if self.num_steps_taken % self.episode_length == 0:
            
            if self.episode_num%2==0:
                 self.left_environment=True
                 print('Back to a right biased policy')
            
            if self.episode_num%4==0 and self.episode_length<400:
                self.left_environment=False
                print('Change to a left biased policy')
         
            if self.episode_num==self.episodes_of_bias_random_exploration:
                #Disable the bias random exploration
                self.random_exploration=False
            if check_progress:
                print('Episode ' + str(self.episode_num) + ', Loss = ' + str(self.av_loss)+',  Epsilon='+str(self.epsilon)+', Episode_length:'+str(self.episode_length))
            
            #+1 episode
            if self.episode_length>1000 and self.episode_num>3:
                self.episode_length-=1000
            elif self.episode_length>200:
               
                self.episode_length-=100
            
            elif self.episode_length==200:
                
                
                self.episode_length-=10
            
           
                
        
            self.episode_num+=1
            self.total_reward=0
            self.num_steps_taken = 0
            #set epsilon value each episode
            self.epsilon_decay()
            return True  
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # decide the next_action in a discrete manner(up,down,right,left)
        
        if self.num_steps_taken<99:#Evaluating greedy policy
           
            discrete_action=self.get_greedy_action_inner(state)
            
        elif self.num_steps_taken==99:
            discrete_action=self.get_greedy_action_inner(state)
            print('Distance to goal according to t-net policy: '+str(self.distance_to_goal))
            if self.distance_to_goal<0.09:#if greedy policy is enough close stop training
                self.random_exploration=False
                   
                self.decay_epsilon=1
                self.epsilon=0
                self.episode_length=100
                if self.distance_to_goal<0.03:
                    self.t_training=False
                    print('Finished training')
            elif self.episode_length<200:
                
                self.epsilon=0.9
                self.decay_epsilon=0.95
                self.episode_length=300
               
                self.random_exploration=True
        elif self.random_exploration:
             discrete_action=self._choose_next_action()
        else:
            state_input_tensor = torch.FloatTensor(state)
            q_values=self.dqn.q_network.forward(state_input_tensor).detach()
            discrete_action=self._choose_next_action_epsilon_greedy(q_values=q_values,epsilon=self.epsilon)
        # Convert the discrete action into a continuous action.
        action = self._discrete_action_to_continuous(discrete_action,self.mov_magnitude)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        # Store the discrete_action for the model
        self.discrete_action=discrete_action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        #Scale the rewards:
        reward =  (1-distance_to_goal)
        if self.state[0]==next_state[0] and self.state[1]==next_state[1] :
            reward=-(reward+self.penalisation_obstacles)
        if distance_to_goal<0.05:
            
            reward =1+0.2-distance_to_goal
        if distance_to_goal<0.25:
            self.left_environment=True*self.left_environment #remain as it is if its true and change if diff
        self.distance_to_goal=distance_to_goal
        self.total_reward+=reward
        # Create a transition
        transition = (self.state, self.discrete_action, reward, next_state)
        # Store the transition in the buffer and if the buffer is enough full start training the network
        self.buffer._append(transition)
        # Start training the network only when we have twice the minibatch_size in our buffer
        buffer_size=len(self.buffer.buffer_reward)
        if buffer_size>=self.buffer.maxlen/5:#Do not start to train unless the buffer is sufficiently full
            # if self.reached_goal==False:
                loss=self.dqn.train_q_network(self.buffer,episode=self.episode_num,step=self.num_steps_taken,minibatch_size=self.minibatch_size,t_training=self.t_training)
                self.av_loss+=loss/self.episode_length
            # else:
            #     self.dqn.update_t_network()
              
       
            
            
        

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        state_input_tensor = torch.FloatTensor(state)
        q_values=self.dqn.t_network.forward(state_input_tensor).detach()
        discrete_action=self._choose_next_action_greedy(q_values=q_values,epsilon=0)#Which means a completely greedy policy
        # Convert the discrete action into a continuous action.
        action = self._discrete_action_to_continuous(discrete_action)
        return action
    
    def get_greedy_action_inner(self, state):
        state_input_tensor = torch.FloatTensor(state)
        q_values=self.dqn.t_network.forward(state_input_tensor).detach()
        discrete_action=self._choose_next_action_greedy(q_values=q_values,epsilon=0)#Which means a completely greedy policy
        # Convert the discrete action into a continuous action.
        
        return discrete_action

    
    ###Added content:
    #Function to decay epsilon
    def epsilon_decay(self):
        self.epsilon=self.epsilon*self.decay_epsilon
        
        
    # Function to convert discrete action(used by a DQN) to a continuous action ( used by the environment).
    def _discrete_action_to_continuous(self, discrete_action,mov_magnitude=0.02):
        #mov_magnitude: represents the step taken by the agent, maximum movement would 0.02.
        # mov_diag=(mov_magnitude/2)
        if discrete_action == 0:#RIGHT
            # Move x to the right, and 0 upwards
            continuous_action = np.array([mov_magnitude, 0], dtype=np.float32)
        elif discrete_action==1:#UP
             # Move 0 to the right, and x upwards
            continuous_action= np.array([0,mov_magnitude],dtype=np.float32)
        elif discrete_action==2:#LEFT
            continuous_action=np.array([-mov_magnitude,0], dtype=np.float32)
        elif discrete_action==3:#DOWN
            continuous_action=np.array([0,-mov_magnitude],dtype=np.float32)
         
        return continuous_action

    # Function to choose the action according to an epsilon-greedy policy
    def _choose_next_action_epsilon_greedy(self,q_values,epsilon):
        #q_values:predicted q_values by the t-network for the current_state
        #epsilon:grade of randomized actions float from [0,1], 1 totally random, 0 totally greedy
        others=epsilon/4#Divide it between the number of options if diagonal movs included should be set to 8
        
        prob=1-epsilon+others
        
        if epsilon==0:
            max_Q=q_values.max()
            A_star=0
            #probs=np.ones_like((1,1,1,1,1,1,1,1))*others #Diagonal case
            probs=np.ones_like((1,1,1,1))*others
            probs=np.nan_to_num(probs)
            for m in range(0,4):#set to 8 if diag movs allowed
                if q_values[m]==max_Q:
                    A_star=m
                    
                    probs[A_star]=1
            action=np.random.choice(range(0,4),1,p=probs/probs.sum())#set to 8 if diag mov allowed
        else:
            max_Q=q_values.max()
            A_star=0
            #probs=np.ones_like((1,1,1,1,1,1,1,1))*others #Diagonal case
            probs=np.ones_like((1,1,1,1))*others
            probs=np.nan_to_num(probs)
            for m in range(0,4):#set to 8 if diag movs allowed
                if q_values[m]==max_Q:
                    A_star=m
                    
                    probs[A_star]=prob
            action=np.random.choice(range(0,4),1,p=probs/probs.sum())#set to 8 if diag mov allowed
        return action
    
    
     # Function to choose the action according to a greedy policy
    def _choose_next_action_greedy(self,q_values,epsilon):
        #q_values:predicted q_values by the t-network for the current_state
        #epsilon:grade of randomized actions float from [0,1], 1 totally random, 0 totally greed
        action=self._choose_next_action_epsilon_greedy(q_values,epsilon)
 
        return action
     # Function for the agent to randomly choose its next action biased against the left side
    def _choose_next_action(self):
        # Return discrete action 0
        if self.left_environment:
            action=np.random.choice(range(0,4),1,p=(0.3,0.3,0.1,0.3))
        else:
            action=np.random.choice(range(0,4),1,p=(0.1,0.3,0.3,0.3))
            #Our initial "random" is biased not to go left
        return action


    # The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        self.input_dim = input_dimension
        self.output_dim = output_dimension
        
        self.common_layer = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU()
        )

        self.q_value_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.state_adv_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_dim)
        )
    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        features = self.common_layer(input)
        values = self.q_value_layer(features)
        advantages = self.state_adv_layer(features)
        output = values + (advantages - advantages.mean())
        
        return output


# The DQN class determines how to train the above neural network.
class DQN:
    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)#set to 8 if diag mov allowed
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=3e-3)
        self.t_network= Network(input_dimension=2, output_dimension=4)#set to 8 if diag mov allowed
        self.t_network.load_state_dict(self.q_network.state_dict())
    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, buffer,episode,step,minibatch=True,target_update=1,minibatch_size=100,prioritize=False,t_training=True):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(buffer,minibatch=minibatch,minibatch_size=minibatch_size,prioritize=prioritize)
        loss.backward()
        
        for param in self.q_network.parameters():#avoid instabilities due to high gradients 
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        # Get the loss as a scalar value
        loss_value = loss.item()
        # Update the t network after the first step every x episodes
        if episode%target_update==0 and step==1 and t_training:
            self.update_t_network()
            
        return loss_value
    
    def update_t_network(self):
        
        self.t_network.load_state_dict(self.q_network.state_dict())
        print("The t_network has been updated")
        
    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self,buffer,minibatch=True,gamma=0.999,minibatch_size=100,prioritize=False):
        
        if minibatch==True:
            minibatch_index=buffer._getminibatch(length=minibatch_size,prioritize=prioritize)
            rewards=list(map(buffer.buffer_reward.__getitem__, minibatch_index))
            reward_batch = torch.FloatTensor(rewards)
            actions=list(map(buffer.buffer_action.__getitem__,minibatch_index))
            action_tensor=torch.LongTensor(actions)
            next_state=list(map(buffer.buffer_nextstate.__getitem__,minibatch_index))
            next_state_input = torch.FloatTensor(next_state)
            minibatch_input=list(map(buffer.buffer_input.__getitem__,minibatch_index))
            minibatch_input_tensor = torch.FloatTensor(minibatch_input)
            
            network_prediction = self.q_network.forward(minibatch_input_tensor).gather(dim=1, index=action_tensor)
            #next_state_values = torch.zeros(100)
            state_qvalues = self.t_network.forward(next_state_input).detach()
            next_state_values=state_qvalues.max(1)[0]
            
            minibatch_labels_tensor = reward_batch+gamma*next_state_values
      
        # Compute the loss based on the label's batch
        loss = torch.nn.SmoothL1Loss()(network_prediction, minibatch_labels_tensor.unsqueeze(1))
        return loss
    
    def _predict_qvalues(self,inputs):
        q_value=self.q_network.forward(inputs)
        return q_value
    
    def create_state_action_matrix(self):
        q_values=np.zeros((50,50,4))
        for col in range(50):
            for row in range(50):
              
                    inputs=[row/50.0,col/50.0]
                    inputs=torch.FloatTensor(inputs)
                    q_values[row,col,:]=self.q_network(torch.tensor(inputs)).detach()
                    
        return q_values
    
    def greedy_policy(self,q_values):
        greedy_pol=np.zeros((50,50))
        for col in range(50):
            for row in range(50):
                    greedy_pol[row,col]=np.argmax(q_values[row,col,:])
        return greedy_pol
    
class ReplayBuffer(object):
    

      def __init__(self,capacity=10000):
         self.buffer_reward=collections.deque(maxlen=capacity)
         self.buffer_input=collections.deque(maxlen=capacity)
         self.buffer_action=collections.deque(maxlen=capacity)
         self.buffer_nextstate=collections.deque(maxlen=capacity)
         self.maxlen=capacity
         self.probabilities=collections.deque(maxlen=capacity)
         self.sampling_weights=collections.deque(maxlen=capacity)
         self.beta=0.2
         self.alfa=0.5
      def _append(self,transition):
        
        N=len(self.buffer_reward)
         
        if self.maxlen>N:
             self.buffer_reward.append(transition[2])
             self.buffer_input.append(transition[0])
             self.buffer_action.append(transition[1])
             self.buffer_nextstate.append(transition[3])
             #Prioritize new sample max.priority=1
             
            
        else:#Different approach if the buffer is full ¿?
           
             self.buffer_reward.append(transition[2])
             self.buffer_input.append(transition[0])
             self.buffer_action.append(transition[1])
             self.buffer_nextstate.append(transition[3])
             
        
      def _getminibatch(self,length,prioritize=False):
          
        selection=np.random.choice(range(len(self.buffer_reward)),length)
        return selection
       
    

       
