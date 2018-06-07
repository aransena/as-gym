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
    def __init__(self, state_bins, action_bins, action_type='', epsilon=0.1, alpha=0.1, gamma=1.0, init_vals=0, plotting=False, plot_params=[0, 1, 0, 1]):
        self._state_bins = state_bins
        self._action_bins = action_bins
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._reward_total = 0.0

        self._action_type = action_type

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

        print "Problem state space {}".format(np.size(self._q_table))
        self._plotting = plotting

        if self._plotting:
            plt.ion()
            self._fig = plt.figure()
            self._nbPlotPoints = 1000

            x = np.linspace(0, self._nbPlotPoints - 1, self._nbPlotPoints)
            y = [0] * self._nbPlotPoints

            self._reward_subplot = self._fig.add_subplot(211)
            plt.axis([0, self._nbPlotPoints, plot_params[0], plot_params[1]])

            plt.xticks(np.array(range(0, self._nbPlotPoints, 200)),
                       map(lambda x: str(x), range(-self._nbPlotPoints, -1, 200)))

            self._params_subplot = self._fig.add_subplot(212)
            plt.axis([0, self._nbPlotPoints, plot_params[2], plot_params[3]])
            plt.xticks(np.array(range(0, self._nbPlotPoints, 200)),
                       map(lambda x: str(x), range(-self._nbPlotPoints, -1, 200)))

            self._rewards_plot = self._reward_subplot.plot(x, y, 'bo')[0]
            self._mean_plot = self._reward_subplot.plot(x, y, 'r-')[0]
            self._median_plot = self._reward_subplot.plot(x, y, 'g-')[0]

            self._alpha_plot = self._params_subplot.plot(x, y, 'r-')[0]
            self._epsilon_plot = self._params_subplot.plot(x, y, 'g-')[0]
            self._gamma_plot = self._params_subplot.plot(x, y, 'b-')[0]

            self._reward_data = [0] * self._nbPlotPoints
            self._mean_data = [0] * self._nbPlotPoints
            self._median_data = [0] * self._nbPlotPoints

            self._epsilon_data = [0] * self._nbPlotPoints
            self._alpha_data = [0] * self._nbPlotPoints
            self._gamma_data = [0] * self._nbPlotPoints

            self.update_plot()

    def _digitize_state(self, state_observation):
        # digitized state returned as tuple to allow easy indexing of q_table
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
        action = None
        if greedy and state_observation is not None:
            action = self._max_action(state_observation)
        else:
            if np.random.random() < self._epsilon or state_observation is None:
                action = np.random.choice(self._action_bins)
            else:
                action = self._max_action(state_observation)

        if "Discrete" in self._action_type:
            return action
        elif "Box" in self._action_type:
            return [action]
        else:
            return action

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
        self._gamma_data.pop(0)
        self._gamma_data.append(self._gamma)

    def update_plot(self, episodes=0):
        min_reward = np.min(self._reward_data[-100:-1])
        max_reward = np.max(self._reward_data[-100:-1])
        print("min reward {}, max reward {}, mean reward {}, median {}".format(min_reward,
                                                                               max_reward,
                                                                               self._mean_data[-1],
                                                                               self._median_data[-1]))

        self._rewards_plot.set_ydata(np.asarray(self._reward_data))
        self._mean_plot.set_ydata(np.asarray(self._mean_data))
        self._median_plot.set_ydata(np.asarray(self._median_data))
        self._alpha_plot.set_ydata(np.asarray(self._alpha_data))
        self._epsilon_plot.set_ydata(np.asarray(self._epsilon_data))
        self._gamma_plot.set_ydata(np.asarray(self._gamma_data))

        # if episodes > self._nbPlotPoints:
        #     x = np.linspace(0, self._nbPlotPoints - 1, self._nbPlotPoints)
        #     x = x + episodes
        #     # self._reward_subplot.set_xlim(x)
        #
        #     self._rewards_plot.set_xdata(x)
        #     self._mean_plot.set_xdata(x)
        #     self._median_plot.set_xdata(x)
        #     # self._alpha_plot.set_xdata(x + episodes)
        #     # self._epsilon_plot.set_xdata(x + episodes)
            # self._gamma_plot.set_xdata(x + episodes)

        self._fig.canvas.draw()
        plt.pause(0.0001)

    def save_q_table(self, filename='q_table.pkl'):
        output_file = open(filename, 'wb')
        pickle.dump(self._q_table, output_file)
