#!/usr/bin/env python

import gym


if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')

    while True:
        env.reset()
        for t in range(1000):
            env.render()
            next_state_observation, reward, done, info = env.step(env.action_space.sample())
            print(next_state_observation)

            if done:
                break
