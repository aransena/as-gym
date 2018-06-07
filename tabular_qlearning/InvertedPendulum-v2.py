#!/usr/bin/env python
"""
Naive approach to InvertedPendulum-v2.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import numpy as np
import gym
from TabularQLearner import TabularQLearner


if __name__ == '__main__':  # Test run for class
    check_interval = 500

    x_bins = np.arange(-2.5, 3.3, 0.8)
    x_dot_bins = np.arange(-4.0, 4.08, 0.08)
    theta_bins = np.arange(-42.2, 42.28, 0.08)
    theta_dot_bins = np.arange(-4.0, 4.02, 0.02)

    state_bins = [x_bins, x_dot_bins, theta_bins, theta_dot_bins]
    action_bins = [0, 1]

    TQL = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[0, 300, 0, 1])
    TQL.set_gamma(1.0)
    env = gym.make('InvertedPendulum-v2')
    print env.action_space, env.observation_space

    episodes = 0

    while True:
        observation = env.reset()
        state_observation = None
        next_state_observation = None
        inspection_run = False

        for t in range(1000):  # episode stops after 200 iterations when done signal is returned below

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
                TQL.update_plot_data()
                if inspection_run:
                    print("Episode finished after {} timesteps, episode {}".format(t + 1, episodes))
                    TQL.update_plot(episodes)
                break

            state_observation = next_state_observation

        episodes = episodes + 1

