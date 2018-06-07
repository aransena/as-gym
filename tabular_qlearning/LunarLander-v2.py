#!/usr/bin/env python
"""
Naive approach to BipedalWalker-v2.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import gym
import numpy as np
from TabularQLearner import TabularQLearner


if __name__ == '__main__':
    ## Setup gym
    env = gym.make('LunarLander-v2')

    ## Tabular Q Learner Setup
    d0 = np.arange(-1.0, 1.5, 0.5)
    a1 = [0, 1, 2, 3]

    state_bins = [d0]*8

    TQL1 = TabularQLearner(state_bins, a1, action_type=str(env.action_space),
                          init_vals=0, plotting=True, plot_params=[-600, 600, 0, 1])
    TQL1.set_gamma(1.0)
    
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

            TQL1.set_epsilon(epsilon)

            a1 = TQL1.get_action(state_observation)  # returns random action if no observation passed

            next_state_observation, reward, done, info = env.step(a1)

            if state_observation is not None and next_state_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-2)))
                TQL1.set_alpha(alpha)
                TQL1.update_q(next_state_observation, state_observation, a1, reward)

            if done:
                TQL1.update_plot_data()

                if inspection_run:
                    print("Episode finished after {} timesteps, episode {}".format(t + 1, episodes))
                    TQL1.update_plot(episodes)
                    check_interval = max(100, check_interval - 1000)

                break

            state_observation = next_state_observation

        episodes = episodes + 1
