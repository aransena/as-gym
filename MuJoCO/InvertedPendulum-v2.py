#!/usr/bin/env python3.6
"""NOTE: python3.6
Naive approach to InvertedPendulum-v2.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import numpy as np
import gym
from agents.TabularQLearner import TabularQLearner

if __name__ == '__main__':  # Test run for class

    env = gym.make('InvertedPendulum-v2')
    state_bins = []
    state_step = 0.2

    print(env.observation_space.high, env.observation_space.low, env.action_space.high, env.action_space.low)
    if np.inf in env.observation_space.high:
        env.render()
        env.reset()
        observations = []
        for i in range(0, 10000):
            next_state_observation, reward, done, info = env.step(env.action_space.sample())
            observations.append(next_state_observation)

        # print(np.asarray(observations))

        for col in range(0, np.shape(np.asarray(observations))[1]):

            ob_max = np.max(np.asarray(observations)[:, col])
            ob_min = np.min(np.asarray(observations)[:, col])
            pad = 1.1
            state_bins.append(np.arange(ob_min*pad, ob_max*pad+state_step, state_step))

    action_bins = np.arange(-3.0, 3.5, 0.5)

    TQL = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[0, 300, 0, 1])
    TQL.set_gamma(1.0)

    # for i in range(0,10):
    #     print(env.action_space.sample())
    #
    # quit()

    print(env.action_space, env.observation_space)

    episodes = 0

    check_interval = 5000

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

