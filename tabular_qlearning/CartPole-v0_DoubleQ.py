#!/usr/bin/env python
"""
Naive state space digitization approach to CartPole-v0.
Uses Double Q Learner. Rewards in plots will appear to be lower than standard Q learner, but this is because
each Q learner can only access at most half of the episodes rewards, the total reward is the same as single Q.
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

    TQL1 = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[0, 300, 0, 1])
    TQL1.set_gamma(1.0)
    TQL2 = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[0, 300, 0, 1])
    TQL2.set_gamma(1.0)
    env = gym.make('CartPole-v0')

    episodes = 0

    while True:
        observation = env.reset()
        state_observation = None
        next_state_observation = None
        inspection_run = False
        reward_tot = 0
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

            TQL1.set_epsilon(epsilon)
            TQL2.set_epsilon(epsilon)

            if np.random.random() > 0.5:
                action = TQL1.get_action(state_observation)  # returns random action if no observation passed
                q_id = 1
            else:
                action = TQL2.get_action(state_observation)  # returns random action if no observation passed
                q_id = 2

            next_state_observation, reward, done, info = env.step(action)
            reward_tot = reward_tot + reward

            if state_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-3)))
                if q_id == 2:
                    TQL1.set_alpha(alpha)
                    TQL1.update_q(next_state_observation, state_observation, action, reward)
                else:
                    TQL2.set_alpha(alpha)
                    TQL2.update_q(next_state_observation, state_observation, action, reward)

            if done:
                TQL1.update_plot_data()
                TQL2.update_plot_data()
                if inspection_run:
                    print("Episode finished after {} timesteps, episode {}".format(t + 1, episodes))
                    TQL1.update_plot(episodes)
                    TQL2.update_plot(episodes)
                break

            state_observation = next_state_observation
        episodes = episodes + 1

