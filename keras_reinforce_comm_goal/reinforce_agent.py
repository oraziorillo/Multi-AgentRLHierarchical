import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import Adam

class ReinforceAgent(object):
    
    def __init__(self, name, n_obs, action_space, policy_learning_rate, value_learning_rate, 
                 discount, baseline=None, n_layers=3, n_neurons=32, restore=False):

        self.name = name
        
        # We need the state and action dimensions to build the network
        self.n_obs = n_obs  
        self.n_act = action_space.n
        
        self.plr = policy_learning_rate
        self.vlr = value_learning_rate
        self.gamma = discount
        self.n_layers = n_layers
        self.n_neurons = n_neurons

        self.use_baseline = baseline is not None
        self.use_adaptive_baseline = baseline == 'adaptive'

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

        policy_loss = keras.losses.sparse_categorical_crossentropy
        policy_optimizer = Adam(lr=self.plr)
        
        policy_layers = [Dense(self.n_neurons, activation='tanh') for _ in range(self.n_layers)]
        policy_layers.append(Dense(self.n_act, activation='softmax'))
        self.policy_net = Sequential(policy_layers)
        self.policy_net.compile(loss=policy_loss, optimizer=policy_optimizer)

        if self.use_baseline and self.use_adaptive_baseline:
            value_loss = keras.losses.mean_squared_error
            value_optimizer = Adam(lr=self.vlr)
            value_layers = [Dense(self.n_neurons, activation='tanh') for _ in range(self.n_layers)]
            value_layers.append(Dense(1))
            self.value_net = Sequential(value_layers)
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

        # Sample action from distribution
        chosen_action = np.random.choice(self.n_act, p=network_output)
        
        return chosen_action

    def _get_returns(self):
        """ This function should process self.episode_rewards and return the discounted episode returns
            at each step in the episode. Hint: work backwards."""
        
        rewards = np.array(self.episode_rewards)

        returns = [rewards[-1]]
        
        for rw in reversed(rewards[:-1]):
            g = rw + returns[-1] * self.gamma
            returns.append(g)    

        return np.array(returns[::-1])
    
    def get_advantages(self):
        """ This function should process self.episode_rewards and return the discounted episode returns
            at each step in the episode. """
        
        returns = self._get_returns()
        std = np.std(returns)
        
        # baseline
        if self.use_baseline:
            if self.use_adaptive_baseline:
                # adaptive baseline                    
                values = np.squeeze(self.value_net(np.array(self.episode_observations)))
                advantages = returns - values
            else:
                # simple baseline
                advantages = returns - np.mean(returns)
        else:
            advantages = returns

        return advantages / std

    def train(self):
        """ When this function is called, the accumulated episode observations, actions and discounted rewards
            should be fed into the network and used for training. Use the _get_returns function to first turn 
            the episode rewards into discounted returns. 
            Apply simple or adaptive baselines if needed, depending on parameters."""
        advantages = self.get_advantages()
        inputs = np.array(self.episode_observations)

        targets = np.array(self.episode_actions)
        batch_size = targets.shape[0]
                
        self.policy_net.train_on_batch(
            inputs, targets, sample_weight=advantages
        )
        
        if self.use_baseline and self.use_adaptive_baseline:
            self.value_net.train_on_batch(
                inputs, advantages
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