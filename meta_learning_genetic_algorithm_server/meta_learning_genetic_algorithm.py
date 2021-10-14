import json
import pickle
import datetime
import pprint
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import genetic as ga
from copy import deepcopy
from ray import tune
from IPython.display import clear_output
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.ppo.ppo import PPOTrainer
from envs.particle_rllib.environment import ParticleEnv
from logger import info_logger, results_logger

"""## Helper functions"""

# Function that creates the environment
def create_env_fn(env_context=None):
    return ParticleEnv(n_listeners=n_listeners,
                       n_landmarks=n_landmarks,
                       render_enable=render_enable)

# Function that maps a policy to its agent id
def policy_mapping_fn(agent_id):
    if agent_id.startswith('manager'):
        return "manager_policy"
    else:
        return "worker_policy"

"""## Parameters"""

# genetic algorithm parameters
n_pop=250
r_cross=0.9
r_mut=0.9

# training parameters
training_algo = "PPO"
env_name = "ParticleManagerListeners"
n_epochs = 10
n_episodes = 3000 # number of episodes in one epoch
n_steps = 25 # number of steps in one episode
learning_rate = 5e-4
tau = 0.01 # for updating the target network
gamma = 0.75 # discount factor
replay_buffer_size = 10000000
batch_size = 1024
hidden_layers = [16, 16]

# environment config parameters
n_listeners = 1
n_landmarks = 12
render_enable = False

# convergence parameters
window_size = 5 # size of the sliding window
min_rel_delta_reward = 0.02  # minimum acceptable variation of the reward

savedata_dir = './savedata/' # savedata directory
checkpoint_dir = './checkpoints/' # checkpoints directory
restore_checkpoint_n = 10

# Create savedata directory
if not os.path.exists(savedata_dir):
    os.makedirs(savedata_dir)

# Create the checkpoint directory
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

"""## Trainers configuration"""

env = create_env_fn()

# According to environment implementation, there exists a different action space and observation space for each agent,
# action_space[0] (resp. observations_space[0]) is allocated for the manager, while the others are allocated for the workers
manager_action_space = env.action_space[0]
manager_observation_space = env.observation_space[0]
worker_action_space = env.action_space[1]
worker_observation_space = env.observation_space[1]

policies = {
    "manager_policy": (None, manager_observation_space, manager_action_space, {"lr": 0.0,}),
    "worker_policy": (None, worker_observation_space, worker_action_space, {"lr": learning_rate,})
    }

training_config = {
    "num_workers": 8,
    "gamma": gamma,
    "horizon": n_steps,
    "train_batch_size": batch_size,
    "model": {
        "fcnet_hiddens": hidden_layers
    },
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": list(policies.keys())
    },
    "no_done_at_end": True,
    "log_level": "ERROR"
}

# Initialize and register the environment
register_env(env_name, create_env_fn)

def objective(i, individual):

    print("Starting evaluation of the individual {}".format(i+1))

    elapsed_episodes = 0
    manager_total_reward = 0

    register_env(env_name, create_env_fn)

    trainer = PPOTrainer(env=env_name, config=training_config)
    weights = trainer.get_weights()
    weights['manager_policy'] = ga.convert_individual_to_manager_weights(individual, weights['manager_policy'])
    trainer.set_weights(weights)

    # Loop for n_episodes
    while elapsed_episodes < n_episodes:
        result = trainer.train()
        elapsed_episodes = result['episodes_total']
        manager_total_reward += (result['policy_reward_mean']['manager_policy'] * result['episodes_this_iter'])
        print(pretty_print(result))

    trainer.stop()
    clear_output()
    return manager_total_reward

def genetic_algorithm(example, n_gen, n_pop, r_cross, r_mut, restore=False):

    if restore:
        with open(checkpoint_dir + "population", 'rb') as fp:
            pop = pickle.load(fp)
        with open(savedata_dir + "epoch-manager-rewards", 'rb') as fp:
            epoch_manager_rewards = pickle.load(fp)
        with open(checkpoint_dir + "best-individual", 'rb') as fp:
            best = pickle.load(fp)
        best_eval = epoch_manager_rewards[-1]
        gen = len(epoch_manager_rewards) + 1

    else:
        # initial population of random bitstring
        pop = [ga.generate_random_individual(example=example) for _ in range(n_pop)]
        # keep track of best solution
        best, best_eval = 0, objective(-1, pop[0])
        # best total reward of the manager per each epoch
        epoch_manager_rewards = []
        gen = 1

    # enumerate generations
    while gen <= n_gen:

        info_logger.info("Current generation: {}".format(gen))

        # evaluate all candidates in the population
        scores = [objective(i, c) for (i, c) in enumerate(pop)]

        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]

        results_logger.info("Generation: {}".format(gen))
        results_logger.info("\tbest score = {:.3f}".format(best_eval))

        # save checkpoint
        with open(checkpoint_dir + "best-individual".format(gen), 'wb') as fp:
            pickle.dump(best, fp)
        info_logger.info("Saved checkpoint after the evaluation of the generation {}".format(gen))

        epoch_manager_rewards.append(best_eval)
        with open(savedata_dir + "epoch-manager-rewards", 'wb') as fp:
            pickle.dump(epoch_manager_rewards, fp)

        plt.plot(epoch_manager_rewards)
        plt.xlabel('Generation')
        plt.ylabel('Reward')
        plt.show()

        # select parents
        selected = [ga.selection(pop, scores, k=(n_pop//10)) for _ in range(n_pop)]

        # create the next generation
        children = list()

        for i in range(0, n_pop, 2):

            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]

            # crossover and mutation
            for c in ga.crossover(p1, p2, r_cross):
                ga.mutation(c, r_mut) # mutation
                children.append(c) # store for next generation

        # replace population
        pop = children

        with open(checkpoint_dir + "population", 'wb') as fp:
            pickle.dump(pop, fp)

        gen += 1

    return [best, best_eval]

ray.init()

trainer = PPOTrainer(env=env_name, config=training_config)

# Print the current configuration
pp = pprint.PrettyPrinter(indent=4)
print("Current configiguration\n-----------------------")
pp.pprint(trainer.get_config())
print("-----------------------\n")

manager_weights_ex = trainer.get_weights()['manager_policy']
trainer.stop()

best, best_eval = genetic_algorithm(example=manager_weights_ex,
                                    n_gen=n_epochs,
                                    n_pop=n_pop,
                                    r_cross=r_cross,
                                    r_mut=r_mut,
                                    restore=False)

ray.shutdown()
