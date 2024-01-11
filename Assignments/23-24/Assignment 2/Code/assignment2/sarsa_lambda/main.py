import numpy as np
import gymnasium as gym
from statistics import mean

from student import sarsa_lambda


def evaluate(num_episodes, render):
    env_name = "Taxi-v3"
    env = gym.make(env_name, render_mode="ansi")
    env_render = gym.make(env_name, render_mode="human" if render else "ansi")

    Q = sarsa_lambda(env)
    rewards = []
    for ep in range(num_episodes):
        tot_reward = 0
        done = False
        s, _ = env_render.reset()
        while not done:
            a = np.argmax(Q[s])
            s, r, done, _, _ = env_render.step(a)
            tot_reward += r
        print("\tTotal Reward ep {}: {}".format(ep, tot_reward))
        rewards.append(tot_reward)
    return mean(rewards)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    render = args.render

    np.random.seed(0)
    import random
    random.seed(0)
    num_episodes = 10
    mean_rew = evaluate(num_episodes, render)
    print("Mean reward over {} episodes: {}".format(num_episodes, mean_rew))
