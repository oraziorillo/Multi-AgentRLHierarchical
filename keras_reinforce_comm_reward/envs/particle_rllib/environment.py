import gym
import logging
from gym import spaces
import numpy as np
from logger import env_logger
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from envs.particle_rllib.core import World, Agent, Landmark, colors

def make_world(n_listeners, n_landmarks, collide):
    world = World()
    # set any world properties first
    world.dim_c = 1
    # add agents
    world.agents = [Agent() for i in range(1 + n_listeners)]
    for i, agent in enumerate(world.agents):
        if i == 0:
            agent.name = 'manager_agent'
        else:
            agent.name = 'worker_agent_%d' % i
        agent.collide = collide
        agent.size = 0.06
    # speaker
    world.agents[0].movable = False
    # listener
    for i in range(1, n_listeners + 1):
        world.agents[i].silent = True
    # add landmarks
    world.landmarks = [Landmark() for i in range(n_landmarks)]
    for i, landmark in enumerate(world.landmarks):
        landmark.name = 'landmark_%d' % i
        landmark.collide = collide
        landmark.movable = False
        landmark.size = 0.04
    # make initial conditions
    reset_world(world)
    return world

def reset_world(world):
    # assign goals to agents
    for agent in world.agents:
        agent.goal = None
    world.agents[0].goal = {}
    # want listeners to go to the goal landmark
    for i in range(1, len(world.agents)):
        world.agents[0].goal['worker_agent_{}'.format(i)] = {'agent': world.agents[i], 'landmark': np.random.choice(world.landmarks)}
    # assign color to manager
    world.agents[0].color = np.array([0.25,0.25,0.25])
    # random properties for landmarks
    n_landmarks = len(world.landmarks)
    for i in range(n_landmarks):
        world.landmarks[i].color = colors[i]
        world.landmarks[i].index = i
    # special colors for workers
    for goal in world.agents[0].goal.values():
        goal['agent'].color = goal['landmark'].color + np.array([0.45, 0.45, 0.45])
    # set random initial states
    for agent in world.agents:
        agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)
    for i, landmark in enumerate(world.landmarks):
        landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)

# environment for all agents in the multiagent world
class ParticleEnv(MultiAgentEnv):

    def __init__(self, n_listeners=1, n_landmarks=3, collide=False, render_enable=False):

        self.world = make_world(n_listeners, n_landmarks, collide)
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(self.world.policy_agents)
        self.render_enable = render_enable

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.world.agents:
            if agent.movable:
                # physical action space
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(self.world.dim_p,), dtype=np.float32)
                self.action_space.append(u_action_space)
            if not agent.silent:
                # communication action space
                c_action_space = spaces.Box(low=-50.0, high=+50.0, shape=(self.world.dim_c,), dtype=np.float32)
                self.action_space.append(c_action_space)
            # observation space
            obs_dim = len(self._obs(agent))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
           

    # step function called by the trainer
    def step(self, action):

        env_logger.debug("Action {}".format(action))

        obs = {}
        rewards = {}
        dones = {}
        infos = {}

        # set action for each agent
        for agent in self.world.agents:
            if agent.name in action:
                self._set_action(action[agent.name], agent)
                
        # advance world state
        self.world.step()

        for agent in self.world.agents:

            env_reward = None

            # if the flag update_comm is set, update both manages' and workers' observations
            # otherwise, update workers' observations only
            if agent.name.startswith('manager'):
                obs[agent.name] = self._obs(agent)
                rewards[agent.name] = self._reward(agent)

            else:
                obs[agent.name] = self._obs(agent)
                rewards[agent.name] = obs[agent.name][-1]
            
            env_logger.debug("Observation {name}: {obs}".format(name=agent.name, obs=obs[agent.name] if agent.name in obs.keys() else "-"))

        # the episode terminates only if the manager is done
        done = self._done()

        return obs, rewards, done, {}
        
    def reset(self):
        # reset world
        reset_world(self.world)
        # record observations for each agent
        obs = {}
        for agent in self.world.agents:
            obs[agent.name] = self._obs(agent)
        return obs

    # get observation for a particular agent
    def _obs(self, agent):
        goals = []
        # goals
        if agent.goal is not None:
            for goal in agent.goal.values():
                goals.append(goal['landmark'].color)

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        for other in self.world.agents:
            if other is agent or other.silent: continue
            comm.append(other.state.c)
            
        # speaker observations
        #   [goal_worker_1, ..., goal_worker_n]
        if not agent.movable:
            return np.concatenate(goals)
        # listener observation:
        #   [x_vel, y_vel, x_entity_1, y_entity_1, ... , x_entity_n, y_entity_n, comm]
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)

    # get reward for a particular agent
    def _reward(self, agent):
        # the env reward is the distance from the listener to landmark
        total_dist = 0
        if not agent.movable:
            for goal in agent.goal.values():
                delta_pos = goal['agent'].state.p_pos - goal['landmark'].state.p_pos
                total_dist += np.sqrt(np.sum(np.square(delta_pos)))
        return - total_dist

    # get dones for a particular agent
    def _done(self):
        done = True
        for goal in self.world.agents[0].goal.values():
            dist_min = goal['agent'].size + goal['landmark'].size
            delta_pos = goal['agent'].state.p_pos - goal['landmark'].state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            done = done and dist < dist_min
        return done

    # set env action for a particular agent
    def _set_action(self, action, agent):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # movement action
            agent.action.u = action
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
        if not agent.silent:
            # communication action
            agent.action.c = action
 