import gym
import logging
from gym import spaces
import numpy as np
from logger import env_logger
from envs.particle_rllib.core import World, Agent, Landmark, colors

def make_world(n_listeners, n_landmarks, collide):
    world = World()
    # set any world properties first
    world.dim_c = n_landmarks
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
        agent.state.c = np.random.randint(0, 11, (1,))
    for i, landmark in enumerate(world.landmarks):
        landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
        landmark.state.p_vel = np.zeros(world.dim_p)

# environment for all agents in the multiagent world
class ParticleEnv():

    def __init__(self, n_listeners=1, n_landmarks=3, collide=False, render_enable=False):

        self.world = make_world(n_listeners, n_landmarks, collide)
        self.agents = self.world.policy_agents
        self.n = len(self.world.policy_agents)
        self.render_enable = render_enable

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.world.agents:
            if agent.movable:
                # physical action space
                u_action_space = spaces.Discrete(self.world.dim_p * 2 + 1)
                self.action_space.append(u_action_space)
            if not agent.silent:
                # communication action space
                c_action_space = spaces.Discrete(self.world.dim_c)
                self.action_space.append(c_action_space)
            # observation space
            obs_dim = len(self._obs(agent))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)
        
        # rendering
        if self.render_enable:
            self.viewer = None
            self._reset_render()    
       

    # step function called by the trainer
    def step(self, action):

        try:
            if(self.render_enable):
                self._reset_render()
                self.render()

            # log communication
            if self.ext_comm_steps == 1:
                env_logger.debug("Manager sets a new goal {}".format(action))
            else:
                env_logger.debug("Manager keeps comminicating the same goal as the previous step {}".format(action))

            # if ext_comm_steps is 8, it's time to update the communication of the goal
            if self.ext_comm_steps == 8:
                self.ext_comm_steps = 1
                obs, rewards, dones, infos = self._exec_step(action, update_comm=True)

            # else, reduce the counter of remaining steps of extended communication
            else:
                self.ext_comm_steps += 1
                obs, rewards, dones, infos = self._exec_step(action, update_comm=False)

            return obs, rewards, dones, infos

        except KeyboardInterrupt as e:
            self.shutdown_viewer()
            raise e
            
    # environment step in which the communication of the manager is updated
    def _exec_step(self, action, update_comm):

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

            # if the flag update_comm is set, update both manages' and workers' observations
            # otherwise, update workers' observations only
            if agent.name.startswith('manager'):
                if update_comm:
                    obs[agent.name] = self._obs(agent)
                rewards[agent.name] = self._reward(agent)

            else:
                obs[agent.name] = self._obs(agent)
                rewards[agent.name] = agent.compute_reward(obs[agent.name])
            
            env_logger.debug("Observation {name}: {obs}".format(name=agent.name, obs=obs[agent.name] if agent.name in obs.keys() else "-"))
        
        done = self._done()

        return obs, rewards, done, {}

    def reset(self):
        # reset world
        reset_world(self.world)
        # extended communication parameter euristic, initially set to 1 to perform the initial communication
        self.ext_comm_steps = 1
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs = {}
        for agent in self.world.agents:
            obs[agent.name] = self._obs(agent)
            env_logger.debug("Observation {name}: {obs}".format(name=agent.name, obs=obs[agent.name] if agent.name in obs.keys() else "-"))
        return obs

    # get observation for a particular agent
    def _obs(self, agent):
        # goal indices
        goal_indices = []
        if agent.goal is not None:
            for goal in agent.goal.values():
                goal_indices.append(goal['landmark'].index)

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        for other in self.world.agents:
            if other is agent or other.silent: continue
            comm.append(other.state.c)

        # speaker observations:
        #   [goal_worker_1, ..., goal_worker_n]
        if not agent.movable:
            return np.array(goal_indices)
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

    # get done
    def _done(self):
        return False

    # set env action for a particular agent
    def _set_action(self, action, agent):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # movement action
            if action == 1: agent.action.u[0] = -1.0
            if action == 2: agent.action.u[0] = +1.0
            if action == 3: agent.action.u[1] = -1.0
            if action == 4: agent.action.u[1] = +1.0
            sensitivity = 3.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
        if not agent.silent:
            # communication action
            agent.action.c = action
            
    # render environment
    def render(self):
        
        if self.viewer == None:
            self.viewer = rendering.Viewer(700,700)
        
        # create rendering geometry
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                if 'agent' in entity.name:
                    geom = rendering.make_circle(entity.size)
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom = rendering.make_square()
                    geom.set_color(*entity.color)
                xform = rendering.Transform()                    
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

        results = []
        # update bounds to center around agent
        cam_range = 1
        pos = np.zeros(self.world.dim_p)
        self.viewer.set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        results.append(self.viewer.render(return_rgb_array = False))

        return results
    
    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
    
    def shutdown_viewer(self):
        self.viewer.close()
