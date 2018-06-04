#!/usr/bin/env python
import gym


if __name__=='__main__':
    # env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v0')
    num_episodes = 20
    for i_episode in xrange(0,num_episodes):
        observation = env.reset()
        tot_reward = 0
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            # if observation[2] < 0.01:
            #     action = 0
            # elif observation[2] > 0.01:
            #     action = 1
            observation, reward, done, info = env.step(action)
            tot_reward = tot_reward + reward
            # print reward
            done = False
            if done:
                print("Episode finished after {} timesteps with reward {}".format(t+1, tot_reward))
                break

