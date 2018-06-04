#!/usr/bin/env python
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

x_bins = np.arange(-1.2, 0.6, 0.1)
x_dot_bins = np.arange(-0.07, 0.07, 0.001)


def QVal(Q_table, observation, action):
    return Q_table[np.digitize(observation[0], x_bins) - 1][np.digitize(observation[1], x_dot_bins) - 1][action]


def QValUpdate(Q_table, observation, action, update):
    Q_table[np.digitize(observation[0], x_bins) - 1][np.digitize(observation[1], x_dot_bins) - 1][action] = update


def MaxQVal(Q_table, observation):
    return np.amax(Q_table[np.digitize(observation[0], x_bins) - 1][np.digitize(observation[1], x_dot_bins) - 1][:])


def MaxAction(Q_table, observation):
    return np.argmax(Q_table[np.digitize(observation[0], x_bins) - 1][np.digitize(observation[1], x_dot_bins) - 1][:])


if __name__=='__main__':
    env = gym.make('MountainCar-v0')
    num_episodes = 10000

    actions = [0, 1, 2]
    Q_table = np.zeros((len(x_bins), len(x_dot_bins), len(actions)))
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

        for t in range(1000):
            if episodes % check_interval == 0:
                epsilon = 0.0
                env.render()
            elif mean_reward < 150.0:
                # epsilon = min(0.9, max(0.0001, np.exp(-(2.0 * episodes / 10000.0))))
                epsilon = 0.1
            else:
                epsilon = 0.0

            # epsilon = min(0.1, max(0.0001, np.exp(-episodes/1000)))

            if np.random.random() < epsilon or previous_observation is None:
                action = env.action_space.sample()  # random action
            else:
                action = MaxAction(Q_table, previous_observation)

            observation, reward, done, info = env.step(action)
            # reward = reward# + observation[0]
            # rewards.pop(0)
            # rewards.append(reward)
            current_observation = observation
            # alpha = min(0.2, max(0.1, np.exp(-mean_reward*2.0/200)))

            if previous_observation is not None:
                update = QVal(Q_table, previous_observation, action) + alpha * (
                            reward + gamma * MaxQVal(Q_table, current_observation) - QVal(Q_table, previous_observation,
                                                                                          action))

                QValUpdate(Q_table, previous_observation, action, update)
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
                if episodes%check_interval == 0:
                    fig.canvas.draw()
                    plt.pause(0.0001)

                break

            previous_observation = current_observation
        episodes = episodes + 1
        #
        if episodes%check_interval == 0:
            epsilon = epsilon_tmp
            print observation, epsilon
            print("Episode finished after {} timesteps, episode {}, mean reward {}, median {}".format(t + 1, episodes,
                                                                                                      mean_data[
                                                                                                          nbPoints - 1], np.median(rewards)))
            # output_file = open('Q_table.pkl', 'wb')
            # pickle.dump(Q_table, output_file)
        if mean_reward > 150.0:
            check_interval = 1
            epsilon = 0.0

    plt.ioff()
    print "Final score: ", t
    print "Num episodes: ", episodes

