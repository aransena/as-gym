#!/usr/bin/env python
"""
Naive approach to CartPole-v0.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import numpy as np
import gym
from TabularQLearner import TabularQLearner


if __name__ == '__main__':  # Test run for class

    o1 = np.arange(-1.0, 1.2, 0.2)
    o2 = np.arange(-1.0, 1.2, 0.2)
    o3 = np.arange(-1.0, 1.2, 0.2)
    o4 = np.arange(-1.0, 1.2, 0.2)
    o5 = np.arange(-20.0, 21.0, 1.0)
    o6 = np.arange(-20.0, 21.0, 1.0)

    state_bins = [o1, o2, o3, o4, o5, o6]
    action_bins = [0, 1, 2]

    TQL = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[0, -500, 0, 1])
    TQL.set_gamma(1.0)
    env = gym.make('Acrobot-v1')
    print env.observation_space, env.action_space

    episodes = 0
    check_interval = 500
    while True:
        observation = env.reset()
        state_observation = None
        next_state_observation = None
        inspection_run = False

        for t in range(500):

            if episodes % check_interval == 0:
                inspection_run = True
                env.render()
            else:
                inspection_run = False

            if inspection_run:
                epsilon = 0.0
                env.render()
            else:
                epsilon = max(0.01, min(0.5, 1 / (episodes * 1e-3)))

            TQL.set_epsilon(epsilon)

            action = TQL.get_action(state_observation)  # returns random action if no observation passed

            next_state_observation, reward, done, info = env.step(action)

            if state_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-3)))
                TQL.set_alpha(alpha)
                TQL.update_q(next_state_observation, state_observation, action, reward)

            if done:
                break

            state_observation = next_state_observation

        TQL.update_plot_data()
        if inspection_run:
            print("Episode finished after {} timesteps, episode {}".format(t + 1, episodes))
            TQL.update_plot(episodes)

        episodes = episodes + 1

