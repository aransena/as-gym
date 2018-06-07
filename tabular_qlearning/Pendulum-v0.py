#!/usr/bin/env python
"""
Naive approach to Pendulum-v0.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import gym
import numpy as np
from TabularQLearner import TabularQLearner


if __name__ == '__main__':
    ## Setup gym
    env = gym.make('Pendulum-v0')

    ## Tabular Q Learner Setup
    costheta_bins = np.arange(-1.0, 1.2, 0.2)
    sintheta_bins = np.arange(-1.0, 1.2, 0.2)
    theta_dot_bins = np.arange(-8.0, 8.8, 0.8)

    state_bins = [costheta_bins, sintheta_bins, theta_dot_bins]
    action_bins = np.arange(-2.0, 2.2, 0.2)

    TQL = TabularQLearner(state_bins, action_bins, action_type=str(env.action_space), init_vals=0, plotting=True, plot_params=[-2000, 0, 0, 1])
    TQL.set_gamma(1.0)

    episodes = 0
    check_interval = 5000

    while True:
        env.reset()
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
                epsilon = max(0.01, min(0.5, 1 / (episodes * 1e-2)))

            TQL.set_epsilon(epsilon)
            action = TQL.get_action(state_observation)  # returns random action if no observation passed

            next_state_observation, reward, done, info = env.step(action)

            if state_observation is not None and next_state_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-2)))
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
