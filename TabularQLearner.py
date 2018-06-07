#!/usr/bin/env python
"""Tabular Q Learning class.

Example:
    # Initialization example
    state_bins = [np.arange(-1.0,1.5,0.5),np.arange(-1.0,1.5,0.5)]
    action_bins = np.arange(0.0,2.0,1.0)
    TQL = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[0, 300, 0, 1])

    # Action selection example
    action = TQL.get_sample_action()  # random action
    action = TQL.get_action(state_observation)

    # Q Val update example
    TQL.update_q(next_state_observation, state_observation, action, reward, alpha, gamma, epsilon)

Contains code to plot reward progress and parameter state.
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle


class TabularQLearner:
    def __init__(self, state_bins, action_bins, epsilon=0.1, alpha=0.1, gamma=1.0, init_vals=0, plotting=False, plot_params=[0, 1, 0, 1]):
        # if callable(epsilon):
        #     self._get_epsilon = epsilon
        # else:
        #     self._get_epsilon = None
        #     self._epsilon = epsilon
        #
        # if callable(alpha):
        #     self._get_alpha = alpha
        # else:
        #     self._get_alpha = None
        #     self._alpha = alpha
        #
        # if callable(gamma):
        #     self._get_gamma = gamma
        # else:
        #     self._get_gamma = None
        #     self._gamma = gamma

        self._state_bins = state_bins
        self._action_bins = action_bins
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._reward_total = 0.0

        dims = []
        for b in self._state_bins:
            dims.append(len(b))

        dims.append(len(action_bins))

        if init_vals == 0:
            self._q_table = np.zeros(dims)
        elif init_vals == 1:
            self._q_table = np.ones(dims)
        else:
            self._q_table = np.random.random(dims)

        self._plotting = plotting

        if self._plotting:
            plt.ion()
            self._fig = plt.figure()
            self._nbPlotPoints = 1000

            self._reward_plot = self._fig.add_subplot(211)
            plt.axis([0, self._nbPlotPoints, plot_params[0], plot_params[1]])
            self._params_plot = self._fig.add_subplot(212)
            plt.axis([0, self._nbPlotPoints, plot_params[2], plot_params[3]])

            x = np.linspace(0, self._nbPlotPoints - 1, self._nbPlotPoints)
            y = [0] * self._nbPlotPoints

            self._rewards_plot = self._reward_plot.plot(x, y, 'bo')[0]
            self._mean_plot = self._reward_plot.plot(x, y, 'r-')[0]
            self._median_plot = self._reward_plot.plot(x, y, 'g-')[0]
            self._alpha_plot = self._params_plot.plot(x, y, 'g-')[0]
            self._epsilon_plot = self._params_plot.plot(x, y, 'm-')[0]

            self._reward_data = [0] * self._nbPlotPoints
            self._mean_data = [0] * self._nbPlotPoints
            self._median_data = [0] * self._nbPlotPoints
            self._epsilon_data = [0] * self._nbPlotPoints
            self._alpha_data = [0] * self._nbPlotPoints

            self.update_plot()

    def _digitize_state(self, state_observation):
        # digitized state returned as tuple to allow easy indexing below
        state = []
        for i, bin in enumerate(self._state_bins):
            state.append(np.digitize(state_observation[i], bin) - 1)
        return tuple(state)

    def _digitize_action(self, action_val):
        return np.digitize(action_val, self._action_bins) - 1

    def _q_val(self, state_observation, action):
        d_state = self._digitize_state(state_observation)
        d_action = self._digitize_action(action)
        return self._q_table[d_state][d_action]

    def _max_qval(self, state_observation):
        d_state = self._digitize_state(state_observation)
        return np.amax(self._q_table[d_state])

    def _max_action(self, state_observation):
        d_state = self._digitize_state(state_observation)
        return self._action_bins[np.argmax(self._q_table[d_state])]

    def _q_val_update(self, state_observation, action, update):
        d_state = self._digitize_state(state_observation)
        d_action = self._digitize_action(action)
        self._q_table[d_state][d_action] = update

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon

    def set_alpha(self, alpha):
        self._alpha = alpha

    def set_gamma(self, gamma):
        self._gamma = gamma

    def update_q(self, next_state_observation, state_observation, action, reward):
        update = self._q_val(state_observation, action) + self._alpha * (
                reward + self._gamma * self._max_qval(next_state_observation) - self._q_val(state_observation, action))

        self._q_val_update(state_observation, action, update)

        if self._plotting:
            self._reward_total = self._reward_total + reward

    def get_action(self, state_observation=None, greedy=False):
        if greedy and state_observation is not None:
            return self._max_action(state_observation)
        else:
            if np.random.random() < self._epsilon or state_observation is None:
                return np.random.choice(self._action_bins)
            else:
                return self._max_action(state_observation)

    def update_plot_data(self):
        self._reward_data.pop(0)
        self._reward_data.append(self._reward_total)
        self._reward_total = 0.0

        mean_reward = np.mean(self._reward_data[-100:-1])
        median_reward = np.median(self._reward_data[-100:-1])

        self._mean_data.pop(0)
        self._mean_data.append(mean_reward)
        self._median_data.pop(0)
        self._median_data.append(median_reward)
        self._alpha_data.pop(0)
        self._alpha_data.append(self._alpha)
        self._epsilon_data.pop(0)
        self._epsilon_data.append(self._epsilon)

    def update_plot(self):
        print("mean reward {}, median {}".format(self._mean_data[-1], self._median_data[-1]))

        self._rewards_plot.set_ydata(np.asarray(self._reward_data))
        self._mean_plot.set_ydata(np.asarray(self._mean_data))
        self._median_plot.set_ydata(np.asarray(self._median_data))
        self._alpha_plot.set_ydata(np.asarray(self._alpha_data))
        self._epsilon_plot.set_ydata(np.asarray(self._epsilon_data))

        self._fig.canvas.draw()
        plt.pause(0.0001)

    def save_q_table(self, filename='q_table.pkl'):
        output_file = open(filename, 'wb')
        pickle.dump(self._q_table, output_file)


if __name__ == '__main__':  # Test run for class
    check_interval = 1000

    x_bins = np.arange(-2.5, 3.3, 0.8)
    x_dot_bins = np.arange(-4.0, 4.08, 0.08)
    theta_bins = np.arange(-42.2, 42.28, 0.08)
    theta_dot_bins = np.arange(-4.0, 4.02, 0.02)

    state_bins = [x_bins, x_dot_bins, theta_bins, theta_dot_bins]
    action_bins = [0, 1]

    TQL = TabularQLearner(state_bins, action_bins, init_vals=0, plotting=True, plot_params=[0, 300, 0, 1])
    TQL.set_gamma(1.0)
    env = gym.make('CartPole-v0')

    episodes = 0

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

            action = TQL.get_action(state_observation)  # returns random action if observation is None

            next_state_observation, reward, done, info = env.step(action)

            if state_observation is not None and not inspection_run:
                alpha = max(0.01, min(0.3, 1 / (episodes * 1e-3)))
                TQL.set_alpha(alpha)
                TQL.update_q(next_state_observation, state_observation, action, reward)

            if done:
                TQL.update_plot_data()
                if inspection_run:
                    print("Episode finished after {} timesteps, episode {}".format(t + 1, episodes))
                    TQL.update_plot()
                break

            state_observation = next_state_observation

        episodes = episodes + 1

