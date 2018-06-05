#!/usr/bin/env python
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

costheta_bins = np.arange(-1.0, 1.0, 0.2)
sintheta_bins = np.arange(-1.0, 1.0, 0.2)
theta_dot_bins = np.arange(-8.0, 8.0, 0.4)
action_bins = np.arange(-2.0, 2.0, 0.4)

print "State Space: ", len(costheta_bins)*len(sintheta_bins)*len(theta_dot_bins)*len(action_bins)


def QVal(Q_table, observation, action):
    return Q_table[np.digitize(observation[0], costheta_bins) - 1][np.digitize(observation[1], sintheta_bins) - 1][
        np.digitize(observation[2], theta_dot_bins) - 1][action]


def QValUpdate(Q_table, observation, action, update):
    Q_table[np.digitize(observation[0], costheta_bins) - 1][np.digitize(observation[1], sintheta_bins) - 1][
        np.digitize(observation[2], theta_dot_bins) - 1][action] = update


def MaxQVal(Q_table, observation):
    return np.amax(Q_table[np.digitize(observation[0], costheta_bins) - 1][np.digitize(observation[1], sintheta_bins) - 1][
        np.digitize(observation[2], theta_dot_bins) - 1][:])


def MaxAction(Q_table, observation):
    return np.argmax(Q_table[np.digitize(observation[0], costheta_bins) - 1][np.digitize(observation[1], sintheta_bins) - 1][
        np.digitize(observation[2], theta_dot_bins) - 1][:])


if __name__=='__main__':
    env = gym.make('Pendulum-v0')
    num_timesteps = 200

    # Q_table = np.random.random((len(costheta_bins), len(sintheta_bins), len(theta_dot_bins), len(action_bins)))
    # Q_table = np.zeros((len(costheta_bins), len(sintheta_bins), len(theta_dot_bins), len(action_bins)))
    Q_table = np.ones((len(costheta_bins), len(sintheta_bins), len(theta_dot_bins), len(action_bins)))
    epsilon = 0.1
    alpha = 0.1
    gamma = 1.0
    t = 0
    check_interval = 5000
    mean_reward = 0.0

    episodes = 0.0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(211)

    nbPoints = 1000

    plt.axis([0, nbPoints, -1000, 0])
    ax2 = fig.add_subplot(212)
    plt.axis([0, nbPoints, 0, 1])

    x = np.linspace(0, nbPoints-1, nbPoints)
    y = [0]*nbPoints

    line1 = ax.plot(x, y, 'bo')[0]
    line2 = ax.plot(x, y, 'r-')[0]
    line2a = ax.plot(x, y, 'g-')[0]
    line3 = ax2.plot(x, y, 'g-')[0]
    line4 = ax2.plot(x, y, 'm-')[0]
    data = [0]*nbPoints
    mean_data = [0]*nbPoints
    median_data = [0] * nbPoints
    epsilon_data = [0]*nbPoints
    alpha_data = [0]*nbPoints

    rewards = [0.0]*100
    rewards_plot = [0.0] * nbPoints
    median_reward = 10.0
    while True:
        observation = env.reset()
        previous_observation = None
        current_observation = None
        inspection_run = False
        for t in range(num_timesteps):
            if episodes % check_interval == 0:# or (median_reward > -10 and episodes > 100):
                inspection_run = True
                env.render()
            else:
                inspection_run = False

            if inspection_run or (median_reward > -10 and episodes > 100):
                epsilon = 0.0
            else:
                epsilon = 0.1
                # epsilon = max(0.0001, 0.8*np.abs(np.sin(episodes*1e-2)))
                # epsilon = max(0.0001, ((15000 - episodes) / 15000.0) * np.abs(np.sin(episodes * 1e-2)))
                # epsilon = max(0.01, 0.9-(episodes/5000))
                epsilon = max(0.001, min(1.0, 1/(episodes*1e-2)))

            if np.random.random() < epsilon or previous_observation is None:
                action_ind = np.digitize(env.action_space.sample(), action_bins)-1
                action_val = action_bins[action_ind]  # random action
            else:
                action_ind = MaxAction(Q_table, previous_observation)
                action_val = [action_bins[action_ind]]

            observation, reward, done, info = env.step(action_val)
            rewards.pop(0)
            rewards.append(reward)

            current_observation = observation

            if previous_observation is not None and not inspection_run:# and not (median_reward > -10 and episodes > 100):
                # alpha = max(0.0001, alpha-(episodes/10000))
                # alpha = max(0.0001, 0.1*np.abs(np.cos(episodes*1e-2)))
                alpha = max(0.01, min(1.0, 1/(episodes*1e-2)))

                # alpha = max(0.0001, 0.5 * ((5000 - episodes) / 5000.0) * np.abs(np.cos(episodes * 1e-2)))
                try:
                    update = QVal(Q_table, previous_observation, action_ind) + alpha * (
                                reward + gamma * MaxQVal(Q_table, current_observation) - QVal(Q_table, previous_observation,
                                                                                              action_ind))
                except Exception as e:
                    print e
                    print action_ind, previous_observation, current_observation


                QValUpdate(Q_table, previous_observation, action_ind, update)

            previous_observation = current_observation



                # break

        mean_data.pop(0)
        median_data.pop(0)
        rewards_plot.pop( 0)
        rewards_plot.append(np.sum(rewards[-num_timesteps:-1]))
        mean_reward = np.mean(rewards_plot[-100:-1])
        mean_data.append(mean_reward)
        median_reward = np.median(rewards_plot[-100:-1])
        median_data.append(median_reward)

        alpha_data.pop(0)
        alpha_data.append(alpha)
        epsilon_data.pop(0)
        epsilon_data.append(epsilon)

        line1.set_ydata(np.asarray(rewards_plot))
        line2.set_ydata(np.asarray(mean_data))
        line2a.set_ydata(np.asarray(median_data))
        line3.set_ydata(np.asarray(alpha_data))
        line4.set_ydata(np.asarray(epsilon_data))

        if episodes % check_interval == 0:# or (median_reward > -10 and episodes > 100):
            fig.canvas.draw()
            # plt.pause(0.0001)

        episodes = episodes + 1
        #
        if episodes%check_interval == 0:
            print("Episode finished after {} timesteps, episode {}, mean reward {}, median {}".format(t + 1, episodes,
                                                                                                      mean_reward, median_reward))
            # output_file = open('Q_table.pkl', 'wb')
            # pickle.dump(Q_table, output_file)
        # if mean_reward > 150.0:
        #     check_interval = 1
        #     epsilon = 0.0

    # plt.ioff()
    # print "Final score: ", t
    # print "Num episodes: ", episodes

