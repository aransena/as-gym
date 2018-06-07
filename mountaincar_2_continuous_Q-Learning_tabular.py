#!/usr/bin/env python
"""
Naive approach to MountainCarContinuous-v0.
State space is digitized to allow learning with a standard tabular Q-Learner.

More challenging than standard moutaincar due to nature of reward signal. From the github:

"Reward is 100 for reaching the target of the hill on the right hand side, minus the squared sum of actions from start to goal.

This reward function raises an exploration challenge, because if the agent does not reach the target soon enough, it will figure out that it is better not to move, and won't find the target anymore.

Note that this reward is unusual with respect to most published work, where the goal was to reach the target as fast as possible, hence favouring a bang-bang strategy."

https://github.com/openai/gym/wiki/MountainCarContinuous-v0
"""

import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

x_bins = np.arange(-1.2, 0.7, 0.1)
x_dot_bins = np.arange(-0.07, 0.071, 0.001)
action_bins = np.arange(-1.0, 2.0, 1.0)


print "State Space: ", len(x_bins)*len(x_dot_bins)*len(action_bins)

def QVal(Q_table, observation, action):
    return Q_table[np.digitize(observation[0], x_bins) - 1][np.digitize(observation[1], x_dot_bins) - 1][action]


def QValUpdate(Q_table, observation, action, update):
    Q_table[np.digitize(observation[0], x_bins) - 1][np.digitize(observation[1], x_dot_bins) - 1][action] = update


def MaxQVal(Q_table, observation):
    return np.amax(Q_table[np.digitize(observation[0], x_bins) - 1][np.digitize(observation[1], x_dot_bins) - 1][:])


def MaxAction(Q_table, observation):
    return np.argmax(Q_table[np.digitize(observation[0], x_bins) - 1][np.digitize(observation[1], x_dot_bins) - 1][:])


if __name__=='__main__':
    env = gym.make('MountainCarContinuous-v0')
    num_episodes = 10000

    actions = [0, 1, 2]
    Q_table = np.random.random((len(x_bins), len(x_dot_bins), len(action_bins)))
    epsilon = 0.1
    alpha = 0.1
    gamma = 1.0
    t = 0

    episodes = 0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(211)

    nbPoints = 1000

    plt.axis([0, nbPoints, -200, 0])
    ax2 = fig.add_subplot(212)
    plt.axis([0, nbPoints, 0, 1])

    x = np.linspace(0, nbPoints-1, nbPoints)
    y = [0]*nbPoints

    rewards_plot = ax.plot(x, y, 'bo')[0]
    mean_plot = ax.plot(x, y, 'r-')[0]
    median_plot = ax.plot(x, y, 'g-')[0]
    alpha_plot = ax2.plot(x, y, 'g-')[0]
    epsilon_plot = ax2.plot(x, y, 'm-')[0]
    data = [0]*nbPoints
    mean_data = [0]*nbPoints
    median_data = [0] * nbPoints
    epsilon_data = [0]*nbPoints
    alpha_data = [0]*nbPoints

    check_interval = 1000
    epsilon_tmp = 0.0
    mean_reward = 0.0
    rewards = [0.0]*100
    rewards_plot = [0.0] * nbPoints

    while True:

        observation = env.reset()
        previous_observation = None
        current_observation = None
        inspection_run = False
        total_reward = 0.0
        for t in range(200):
            if episodes % check_interval == 0:
                inspection_run = True
                env.render()
            else:
                inspection_run = False

            if inspection_run:
                epsilon = 0.0
                env.render()
            else:
                epsilon = max(0.01, min(0.1, 1 / (episodes * 1e-3)))

            if np.random.random() < epsilon or previous_observation is None:
                sample_action = env.action_space.sample()  # random action
                action_ind = np.digitize(sample_action, action_bins)-1
                action = action_bins[action_ind]
            else:
                action_ind = MaxAction(Q_table, previous_observation)
                action = [action_bins[action_ind]]

            observation, reward, done, info = env.step(action)

            if reward > 0:
                break

            reward = (-1)
            total_reward = total_reward + reward
            current_observation = observation

            if previous_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.1, 1 / (episodes * 1e-3)))
                update = QVal(Q_table, previous_observation, action_ind) + alpha * (
                            reward + gamma * MaxQVal(Q_table, current_observation) - QVal(Q_table, previous_observation,
                                                                                          action_ind))
                QValUpdate(Q_table, previous_observation, action_ind, update)

            previous_observation = current_observation

        rewards.pop(0)
        rewards.append(total_reward)
        rewards_plot.pop(0)
        rewards_plot.append(total_reward)

        mean_data.pop(0)
        mean_reward = np.mean(rewards[-100:-1])
        mean_data.append(mean_reward)

        median_data.pop(0)
        median_reward = np.median(rewards[-100:-1])
        median_data.append(median_reward)

        alpha_data.pop(0)
        alpha_data.append(alpha)
        epsilon_data.pop(0)
        epsilon_data.append(epsilon)

        rewards_plot.set_ydata(np.asarray(rewards_plot))
        mean_plot.set_ydata(np.asarray(mean_data))
        median_plot.set_ydata(np.asarray(median_data))
        alpha_plot.set_ydata(np.asarray(alpha_data))
        epsilon_plot.set_ydata(np.asarray(epsilon_data))

        if inspection_run:

            fig.canvas.draw()
            plt.pause(0.0001)

        episodes = episodes + 1

        if inspection_run:
            print("Episode finished after {} timesteps, episode {}, mean reward {}, median {}".format(t + 1, episodes,
                                                                                                      mean_reward,
                                                                                                      median_reward))
            # output_file = open('Q_table.pkl', 'wb')
            # pickle.dump(Q_table, output_file)
    #
    # plt.ioff()
    # print "Final score: ", t
    # print "Num episodes: ", episodes

