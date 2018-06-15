"""
NOTE: Python 3.6
"""

import gym
import numpy as np
from tools.TileCoding import estimate_observation_space_range
from agents.TabularQLearner import TabularQLearner

if __name__ == '__main__':
    env = gym.make('Reacher-v2')
    observation_space_ranges = estimate_observation_space_range(env)
    state_bins = []

    for r in observation_space_ranges:
        srange = np.sum(np.abs(np.asarray(r)))
        if srange != 0.0:
            step = srange / 4.250
        else:
            step = 1.0

        sbin = np.arange((r[0] - 0.0001) * 1.1, (r[1] + 0.0001) * 1.1 + step, step)
        state_bins.append(sbin)

    action_step = 0.25
    action_bins = np.arange(-1.0, 1.0+action_step, action_step)

    state_bins.append(action_bins)

    TQL1 = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[0, -100, 0, 1])
    TQL1.set_gamma(1.0)
    TQL2 = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=False)
    TQL2.set_gamma(1.0)

    episodes = 0
    check_interval = 200000
    stop = False
    epsilon = 0.0

    while True:
        observation = env.reset()
        state_observation = None
        ob1 = None
        ob2 = None
        action1 = None
        action2 = None
        next_state_observation = None
        inspection_run = False

        for t in range(1000):  # episode stops after 200 iterations when done signal is returned below

            if episodes % check_interval == 0 or stop:
                inspection_run = True
                env.render()
            else:
                inspection_run = False

            if inspection_run:
                epsilon = 0.0
                env.render()
            else:
                epsilon = max(0.01, min(0.5, 1 / (episodes * 1e-3)))

            if state_observation is not None:
                lso = next_state_observation.tolist()
                ob1 = lso[:]
                ob2 = lso[:]
                ob1.append(action2)
                ob2.append(action1)

            TQL1.set_epsilon(epsilon)
            action1 = TQL1.get_action(ob1)  # returns random action if no observation passed
            TQL2.set_epsilon(epsilon)
            action2 = TQL2.get_action(ob2)  # returns random action if no observation passed

            next_state_observation, reward, done, info = env.step([action1, action2])# action2])

            lso = next_state_observation.tolist()
            ob1_n = lso[:]
            ob2_n = lso[:]
            ob1_n.append(action2)
            ob2_n.append(action1)

            if state_observation is not None and not inspection_run and ob1 is not None:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-3)))
                TQL1.set_alpha(alpha)
                TQL1.update_q(ob1_n, ob1, action1, reward)
                TQL2.set_alpha(alpha)
                TQL2.update_q(ob2_n, ob2, action2, reward)

            if done:
                break

            state_observation = next_state_observation

        TQL1.update_plot_data()
        if inspection_run:
            print("Episode finished after {} timesteps, episode {}".format(t + 1, episodes))
            TQL1.update_plot(episodes)
            if episodes > 0.0:
                stop = True

        episodes = episodes + 1

