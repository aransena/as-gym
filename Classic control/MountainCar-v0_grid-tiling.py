#!/usr/bin/env python
"""
Naive approach to MountainCar-v0.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import gym
import numpy as np
from scipy.interpolate import interp1d
from tools.TileCoding import IHT, tiles
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


if __name__ == '__main__':
    # fig, ax = plt.subplots()
    # fig.set_tight_layout(True)
    # anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)

    max_size = 256
    num_tilings = 4

    iht = IHT(max_size)

    QVals = np.zeros((max_size, 3))#np.asarray(np.asarray([0]*1024), np.asarray([0,1,2]))

    env = gym.make('MountainCar-v0')

    d1_scale = interp1d([-1.2, 0.6], [0, 9])
    d2_scale = interp1d([-0.07, 0.07], [0, 9])
    actions = [0, 1, 2]

    episodes = 0

    check_interval = 2000
    rewards =[]
    while True:
        env.reset()
        state_observation = None
        next_state_observation = None
        inspection_run = False

        alpha = 1/(num_tilings*10)
        gamma = 1.0
        epsilon = 0.5
        indices = None
        reward_tot = 0.0
        for t in range(1000):
            epsilon = max(0.01, epsilon - 0.01)
            if episodes%check_interval == 0.0:
                env.render()

            if indices is None:
                action = env.action_space.sample()
            else:
                if epsilon < np.random.random():
                    action = env.action_space.sample()
                else:
                    max_actions = []
                    qvals = []
                    for i in range(0, num_tilings):
                        qvals.append(np.amax(QVals[indices[i], :]))
                        max_actions.append(np.argmax(QVals[indices[i], :]))

                    action = max_actions[np.argmax(qvals)]

            next_state_observation, reward, done, info = env.step(action)
            reward_tot = reward_tot + reward
            if done:
                break

            d1 = float(d1_scale(next_state_observation[0]))
            d2 = float(d2_scale(next_state_observation[1]))
            next_obv = [d1, d2]
            next_indices = tiles(iht, num_tilings, next_obv)

            if epsilon < np.random.random():
                next_action = env.action_space.sample()
            else:
                max_actions = []
                qvals = []
                for i in range(0, num_tilings):
                    qvals.append(np.amax(QVals[next_indices[i], :]))
                    max_actions.append(np.argmax(QVals[next_indices[i], :]))

                next_action = max_actions[np.argmax(qvals)]

            if indices is not None:
                for i in range(0, num_tilings):
                    QVals[indices[i], action] = QVals[indices[i], action] + \
                                                alpha*(reward + gamma*QVals[next_indices[i], next_action] -
                                                       QVals[indices[i], action])
            indices = next_indices
        rewards.append(reward_tot)

        if episodes % check_interval == 0.0:
            plt.figure()
            plt.subplot(4, 1, 1)
            plt.scatter(range(0, len(QVals[:, 0])), QVals[:, 0])
            plt.subplot(4, 1, 2)
            plt.scatter(range(0, len(QVals[:, 1])), QVals[:, 1])
            plt.subplot(4, 1, 3)
            plt.scatter(range(0, len(QVals[:, 2])), QVals[:, 2])
            plt.subplot(4, 1, 4)
            plt.scatter(range(0, len(rewards)), np.asarray(rewards)[:])
            plt.show()
        episodes = episodes + 1
