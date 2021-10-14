import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.evaluation import RolloutWorker
from envs.particle_rllib.environment import colors

# Class defining custom callback functions
class CustomCallbacks(DefaultCallbacks):

    # Callback function - episode start
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):

        episode.user_data["correct_goal"] = []


    # Callback function - episode step
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):

        comm_goal = episode.last_action_for('manager_agent')
        real_goal = episode.last_observation_for('manager_agent')[0]

        episode.user_data["correct_goal"].append(comm_goal == real_goal)

    # Callback function - episode end
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):

        correct_goal = episode.user_data["correct_goal"]
        episode.custom_metrics["prob_correct_goal"] = np.sum(correct_goal) / len(correct_goal)
