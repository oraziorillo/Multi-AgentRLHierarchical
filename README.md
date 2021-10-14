# Multi-agent Meta Reinforcement Learning via Hierarchical Interactions
## Description
This is the semester project I did during my MSc in Computer Science at EPFL. 

Reinforcement learning has proven to be very successful in solving various learning problems, that go from the domain of games to that of robotics. However, the dominant strategy in many practical applications consists in adapting the model to the specific task to be performed.

This project aims to introduce a framework in which agents are able to adapt by themselves to the task. While trying to understand which is the most effective way of doing it, we will explore new possibilities in multi-agent reinforcement learning, in order to find a configuration that can enable an easy adaptation of our model to any task that needs to be solved. By taking a cue from the meta-learning philosophy, we want to train our agents not to perform the task, but to learn how to learn to perform the task. Ultimately, our approach will consist in combining hierarchical reinforcement learning and meta-learning, by meta-training an agent (the ‘manager’) to learn how to communicate a reward function in such a way that the (potentially many) other agents (the ‘workers’) are guided by the former to learn how to execute a task.

To find a more detailed description and the results I obtained refer to the {final report}[] in the docs folder.

## Repository's structure
Each folder contains a jupyter notebook that contains the code that has to be run to train the agents and test them with a specific method. Most of them use techniques that are well explained in the report I have previously mentioned, while some of them are only experiments that we    re not included in the final summary, either because of similarity to other methods or poor results.


The folder is organised as follows.

- Each subfolder (except for meta_learning_genetic_algorithm_server) contains a jupyter notebook that can be run in order to start the experiment. To run the experiments on Google Colab, upload the folder on your drive in the main directory. Then, open the jupyter notebook of the experiment you want to run and run all the cells. The permission to access to your Google Drive account will be requested the first time the notebook is run in a session only. It is possible to change the location of the folder on your drive, but then it is necessary to change the relative path of the folder in the notebook, too.
- Each subfolder contains everything needed in order to run a different experiment.
  - fmh_ppo and fmh_ddpg are the standard FMH implementation (as in the paper). The only difference between the two is the RL algorithm, in one case DDPG and PPO in the other.
  - fmh_comm_reward is an implementation of FMH in which the manager communicates the reward to the worker instead of the goal.
  - keras_reinforce_comm_goal and keras_reinforce_comm_reward are respectively the implementation of standard FMH and the implementation of FMH in which the manager communicates the reward to the worker instead of the goal, both using the RL algorithm REINFORCE.
  - meta_learning_random_search is the implementation of FMH in which the manager is meta trained in an outer loop using random search over a vector of weights.
- meta_learning_genetic_algorithm and meta_learning_genetic_algorithm_server are the same implementation of FMH in which the manager is meta trained using a genetic algorithm as non-gradient-based method. The difference between the two experiments is that the first is designed to run on a jupyter notebook, while the second one is based on a script that can easily be run on a server. Indeed, in the folder you can find the instructions to automatically configure the server with the necessary packages, then simply run the script meta_learning_genetic_algorithm.py while it is in its folder.
  - 'Experiment Descriptions.docx' is a Word document that describes the most significant experiments that we did with these frameworks and that are also mentioned in the report.
  - 'Results Sheet.xlsx' is an Excel document that shows most of the results from the experiments described in 'Experiment Descriptions.docx'.