#!/usr/bin/env python
import numpy as np
import gym
from agents.TabularQLearner import TabularQLearner


if __name__ == '__main__':
    env = gym.make('Breakout-ram-v0')

    while True:
        env.reset()
        for t in range(1000):
            env.render()
            next_state_observation, reward, done, info = env.step(env.action_space.sample())
            print(next_state_observation)

            if done:
                break
