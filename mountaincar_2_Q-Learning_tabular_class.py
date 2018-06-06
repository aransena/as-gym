#!/usr/bin/env python
"""
Naive approach to CartPole-v0.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import gym
import numpy as np
from TabularQLearner import TabularQLearner
import pickle
import matplotlib.pyplot as plt




if __name__ == '__main__':  # Test run for class
    check_interval = 1000

    x_bins = np.arange(-1.2, 0.7, 0.1)
    x_dot_bins = np.arange(-0.07, 0.071, 0.001)

    state_bins = [x_bins, x_dot_bins]
    action_bins = [0, 1, 2]

    TQ = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[-200, 0, 0, 1])

    env = gym.make('MountainCar-v0')

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

            if np.random.random() < epsilon or state_observation is None:
                action = TQ.get_sample_action()  # random action
            else:
                action = TQ.get_action(state_observation)

            next_state_observation, reward, done, info = env.step(action)
            #current_observation = observation

            if state_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-3)))
                gamma = 1.0
                TQ.update_q(next_state_observation, state_observation, action, reward, alpha, gamma, epsilon)

            if done:
                TQ.update_plot_data()
                if inspection_run:
                    print("Episode finished after {} timesteps, episode {}".format(t + 1, episodes))
                    TQ.update_plot()
                break

            state_observation = next_state_observation

        episodes = episodes + 1
