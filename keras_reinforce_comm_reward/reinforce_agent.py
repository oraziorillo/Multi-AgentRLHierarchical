import numpy as np
import keras
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

tf.compat.v1.enable_eager_execution()

class ReinforceAgent(object):
    
    def __init__(self, name, n_obs, action_space, policy_learning_rate, value_learning_rate, 
                 discount, n_layers=3, n_neurons=32, restore=False):
        
        self.name = name

        # We need the state and action dimensions to build the network
        self.n_obs = n_obs  
        self.action_space = action_space
        
        self.plr = policy_learning_rate
        self.vlr = value_learning_rate
        self.gamma = discount
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        #These lists stores the cumulative observations for this episode
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        #Build/restore the keras network
        if restore:
            self.restore() 
        else:
            self._build_network()
        
    def _build_network(self):
        """ This function should build the network that can then be called by decide and train. 
            The network takes observations as inputs and has a policy distribution as output."""

        """ This function should build the network that can then be called by decide and train. 
            The network takes observations as inputs and has a policy distribution as output."""
        
        inputs = Input(shape=(self.n_obs,))
        
        prev_layer = inputs
        for _ in range(self.n_layers):
            curr_layer = Dense(self.n_neurons, activation='relu')(prev_layer)
            prev_layer = curr_layer
        action = Dense(self.action_space.shape[0], activation='sigmoid')(curr_layer)
        policy_loss = keras.losses.mean_absolute_error
        policy_optimizer = Adam(learning_rate=self.plr)
        self.policy_net = Model(inputs=inputs, outputs=action)
        self.policy_net.compile(loss=policy_loss, optimizer=policy_optimizer)

        prev_layer = inputs
        for _ in range(self.n_layers):
            curr_layer = Dense(self.n_neurons, activation='relu')(prev_layer)
            prev_layer = curr_layer
        value = Dense(1, activation='linear')(curr_layer)
        value_loss = keras.losses.mean_squared_error
        value_optimizer = Adam(learning_rate=self.vlr)
        self.value_net = Model(inputs=inputs, outputs=value)
        self.value_net.compile(loss=value_loss, optimizer=value_optimizer)

    def observe(self, state, action, reward):
        """ This function takes the observations the agent received from the environment and stores them
            in the lists above."""
        self.episode_observations.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        
    def decide(self, state):
        """ This function feeds the observed state to the network, which returns a distribution
            over possible actions. Sample an action from the distribution and return it."""
        # Create the input (should be 2-D due to Keras usage)
        batch_input = np.expand_dims(state, 0)
        
        # Compute the network output, unpacking the 2-D result
        (network_output,) = self.policy_net(batch_input).numpy()

        mapped_output = self.action_space.low + network_output * (self.action_space.high - self.action_space.low)
        
        return mapped_output
        
    def _get_returns(self):
        """ This function should process self.episode_rewards and return the discounted episode returns
            at each step in the episode. Hint: work backwards."""
        returns = [self.episode_rewards[-1]]
        
        for rw in reversed(self.episode_rewards[:-1]):
            g = rw + returns[-1] * self.gamma
            returns.append(g)
        
        discounted_returns = np.array(returns[::-1])
        discounted_returns -= np.mean(discounted_returns)
        discounted_returns /= np.std(discounted_returns)

        return discounted_returns

    def train(self):
        """ When this function is called, the accumulated episode observations, actions and discounted rewards
            should be fed into the network and used for training. Use the _get_returns function to first turn 
            the episode rewards into discounted returns. 
            Apply simple or adaptive baselines if needed, depending on parameters."""

        inputs = np.array(self.episode_observations)
        targets = np.array(self.episode_actions)

        returns = self._get_returns()
        baseline = self.value_net(inputs).numpy()[:, 0]
        centered_discounted_returns = returns - baseline
                
        self.policy_net.train_on_batch(
            inputs, targets, sample_weight=centered_discounted_returns
        )
        
        self.value_net.train_on_batch(
            inputs, returns
        )

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []
       
    def save(self):
        models_dir = 'models/'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.policy_net.save(models_dir + self.name + "_policy_net")
        self.value_net.save(models_dir + self.name + "_value_net")

    def restore(self):
        models_dir = 'models/'
        self.policy_net = keras.models.load_model(models_dir + self.name + "_policy_net")        
        self.value_net = keras.models.load_model(models_dir + self.name + "_value_net")        