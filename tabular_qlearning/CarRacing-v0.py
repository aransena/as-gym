#!/usr/bin/env python
"""
Naive approach to CarRacing-v0.
Prohibitively slow on laptop CPU.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import gym
import numpy as np
from TabularQLearner import TabularQLearner


if __name__ == '__main__':
    ## Setup gym
    env = gym.make('CarRacing-v0')
    for i in range(0,10):
        if __name__ == '__main__':
            print env.action_space.sample()
    ## Tabular Q Learner Setup
    d0 = np.arange(0.0, 200.0, 50.0)
    a1 = np.arange(-1.0, 1.5, 0.5)

    state_bins = [d0]*11

    TQL1 = TabularQLearner(state_bins, a1, action_type=str(env.action_space),
                          init_vals=0, plotting=True, plot_params=[-600, 600, 0, 1])
    TQL1.set_gamma(1.0)

    TQL2 = TabularQLearner(state_bins, a1, action_type=str(env.action_space),
                           init_vals=0, plotting=True, plot_params=[-600, 600, 0, 1])
    TQL2.set_gamma(1.0)

    TQL3 = TabularQLearner(state_bins, a1, action_type=str(env.action_space),
                           init_vals=0, plotting=True, plot_params=[-600, 600, 0, 1])
    TQL3.set_gamma(1.0)
    
    episodes = 0
    check_interval = 100

    while True:
        env.reset()
        state_observation = None
        next_state_observation = None
        inspection_run = False
        if episodes%10 == 0:
            print "Episode ", episodes, "..."

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

            TQL1.set_epsilon(epsilon)
            TQL2.set_epsilon(epsilon)
            TQL3.set_epsilon(epsilon)

            a1 = TQL1.get_action(state_observation)  # returns random action if no observation passed
            a2 = TQL2.get_action(state_observation)  # returns random action if no observation passed
            a3 = TQL3.get_action(state_observation)  # returns random action if no observation passed

            action = [a1[0], a2[0], a3[0]]

            observation, reward, done, info = env.step(np.array(action))
            next_state_observation = np.reshape(observation, (1, np.size(observation)))[0]
            next_state_observation = np.convolve([1.0/(27648-10)]*(27648-10), next_state_observation, 'valid')

            if state_observation is not None and next_state_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-2)))
                TQL1.set_alpha(alpha)
                TQL2.set_alpha(alpha)
                TQL3.set_alpha(alpha)

                TQL1.update_q(next_state_observation, state_observation, a1, reward)
                TQL2.update_q(next_state_observation, state_observation, a1, reward)
                TQL3.update_q(next_state_observation, state_observation, a1, reward)

            state_observation = next_state_observation

            if done:
                break

        TQL1.update_plot_data()
        TQL2.update_plot_data()
        TQL3.update_plot_data()

        if inspection_run:
            print("Episode finished after {} timesteps, episode {}".format(t + 1, episodes))
            TQL1.update_plot(episodes)
            TQL2.update_plot(episodes)
            TQL3.update_plot(episodes)
            check_interval = max(100, check_interval - 1000)

        episodes = episodes + 1
