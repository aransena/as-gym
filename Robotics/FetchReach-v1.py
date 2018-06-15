"""
NOTE: Python 3.6
"""

import gym


if __name__ == '__main__':
    env = gym.make('FetchReach-v1')
    env.reset()
    while True:
        env.render()
        next_state_observation, reward, done, info = env.step(env.action_space.sample())
        print(next_state_observation)

        if done:
            quit()
            env.reset()


