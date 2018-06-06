#!/usr/bin/env python
"""
Naive approach to MountainCar-v0.
State space is digitized to allow learning with a standard tabular Q-Learner.
"""

import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

x_bins = np.arange(-1.2, 0.7, 0.1)
x_dot_bins = np.arange(-0.07, 0.071, 0.001)

print "State Space: ", len(x_bins)*len(x_dot_bins)*3  # 3 possible actions


def digitize_state(state_observation):
    # digitized state returned as tuple to allow easy indexing below
    return (np.digitize(state_observation[0], x_bins) - 1, np.digitize(state_observation[1], x_dot_bins) - 1)


def q_val(q_table, state_observation, action):
    d_state = digitize_state(state_observation)
    return q_table[d_state][action]


def q_val_update(q_table, state_observation, action, update):
    d_state = digitize_state(state_observation)
    q_table[d_state][action] = update


def max_qval(q_table, state_observation):
    d_state = digitize_state(state_observation)
    return np.amax(q_table[d_state][:])


def max_action(q_table, state_observation):
    d_state = digitize_state(state_observation)
    return np.argmax(q_table[d_state][:])


if __name__=='__main__':
    env = gym.make('MountainCar-v0')
    num_episodes = 10000

    actions = [0, 1, 2]
    qtable = np.zeros((len(x_bins), len(x_dot_bins), len(actions)))
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

    line1 = ax.plot(x, y, 'bo')[0]
    line2 = ax.plot(x, y, 'r-')[0]
    line3 = ax2.plot(x, y, 'g-')[0]
    line4 = ax2.plot(x, y, 'm-')[0]
    data = [0]*nbPoints
    mean_data = [0]*nbPoints
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

            if np.random.random() < epsilon or previous_observation is None:
                action = env.action_space.sample()  # random action
            else:
                action = max_action(qtable, previous_observation)

            observation, reward, done, info = env.step(action)
            current_observation = observation

            if previous_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-3)))
                update = q_val(qtable, previous_observation, action) + alpha * (
                            reward + gamma * max_qval(qtable, current_observation) - q_val(qtable,
                                                                                           previous_observation,
                                                                                           action))

                q_val_update(qtable, previous_observation, action, update)
            if done:
                rewards.pop(0)
                rewards.append(-(t+1))
                rewards_plot.pop(0)
                rewards_plot.append(-(t + 1))

                mean_data.pop(0)
                mean_reward = np.mean(rewards)
                mean_data.append(mean_reward)
                alpha_data.pop(0)
                alpha_data.append(alpha)
                epsilon_data.pop(0)
                epsilon_data.append(epsilon)

                line1.set_ydata(np.asarray(rewards_plot))
                line2.set_ydata(np.asarray(mean_data))
                line3.set_ydata(np.asarray(alpha_data))
                line4.set_ydata(np.asarray(epsilon_data))

                if inspection_run:
                    fig.canvas.draw()
                    plt.pause(0.0001)

                break

            previous_observation = current_observation
        episodes = episodes + 1

        if inspection_run:
            epsilon = epsilon_tmp

            print("Episode finished after {} timesteps, episode {}, mean reward {}, median {}".format(t + 1, episodes,
                                                                                                      mean_data[
                                                                                                          nbPoints - 1],
                                                                                                      np.median(rewards)))
