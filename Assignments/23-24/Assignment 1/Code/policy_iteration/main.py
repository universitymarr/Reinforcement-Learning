import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from policy_iteration import policy_iteration
import argparse
import random
import numpy as np
random.seed(1) # do not modify
np.random.seed(1)  # do not modify

def main():
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    render = args.render

    # Setting environment attributes
    ENV_SIZE = 5
    START_STATE = (0,0)
    END_STATE = (ENV_SIZE-1, ENV_SIZE-1)
    DIRECTIONS = np.array([
                [0,-1], #LEFT
                [1,0], #DOWN
                [0,1], #RIGHT
                [-1,0], #UP
            ])

    rewards = []
    for i in range(3):
        print(f"Starting game {i+1}")

        # Generating the environment
        env_map = generate_random_map(size=ENV_SIZE, seed=3)
        OBSTACLES = np.zeros((ENV_SIZE,ENV_SIZE))
        for row_idx, row in enumerate(env_map):
            for col_idx, state in enumerate(row):
                if state == 'H':
                    OBSTACLES[row_idx][col_idx] = 1

        env = gym.make('FrozenLake-v1', desc=env_map, is_slippery=True, render_mode='human' if render else 'rgb_array')

        policy, values = policy_iteration(env, ENV_SIZE, END_STATE, DIRECTIONS, OBSTACLES)
        print("Policy:")
        print(policy)
        print("Values:")
        print(values)

        state, _ = env.reset()
        i = state // ENV_SIZE
        j = state % ENV_SIZE
        state = (i,j)
        if render: env.render()

        total_reward = 0.
        done = False
        while not done:
            action = policy[state[0],state[1]]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            i = state // ENV_SIZE
            j = state % ENV_SIZE
            state = (i,j)

            total_reward += reward
            if render: env.render()
        print("\tTotal Reward:", total_reward)
        rewards.append(total_reward)


    print("Mean Reward: ", np.mean(rewards))


if __name__ == '__main__':
    main()