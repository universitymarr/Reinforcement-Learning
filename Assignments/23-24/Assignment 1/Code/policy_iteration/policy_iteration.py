import random
import numpy as np
import gymnasium as gym

from student import *

def policy_iteration(env, env_size, end_state, directions, obstacles, gamma=0.99, max_iters=1000, theta=1e-3):
    # rename to policy
    policy = np.random.randint(0, env.action_space.n, (env.observation_space.n))

    values = np.random.random((env.observation_space.n))
    STATES = np.zeros((env.observation_space.n, 2), dtype=np.uint8)
    REWARDS = reward_probabilities(env_size)

    i = 0
    for r in range(env_size):
        for c in range(env_size):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1

    for i in range(max_iters):

        # policy evaluation
        while True:
          delta = 0
          for s in range(env.observation_space.n):
              state = STATES[s]
              v_old = values[s]
              # deterministic policy, so no need for weighted sum over actions
              a = policy[s]
              next_state_prob = transition_probabilities(env, state, a, env_size, directions, obstacles).flatten()
              done = (state == end_state).all()
              values[s] = (1-done)*(next_state_prob*(REWARDS + gamma*values)).sum()
              delta = max(delta, np.abs(v_old - values[s]))
          if delta < theta:
              break
          
        # policy improvement
        policy_stable = True
        old_policy = policy.copy()
        for s in range(env.observation_space.n):
            state = STATES[s]
            b = policy[s] # save old best action
            # compute new best action based on the updated value
            best_value = -float('inf')
            best_action = None
            for a in range(env.action_space.n):
                next_state_prob = transition_probabilities(env, state, a, env_size, directions, obstacles).flatten()
                va = (next_state_prob*(REWARDS + gamma*values)).sum()
                if va > best_value:
                    best_value = va
                    best_action = a
            policy[s] = best_action
            if best_action != b:
                policy_stable = False

        if policy_stable:
            break

    print(f'finished in {i+1} iterations')
    
    return policy.reshape((env_size, env_size)), values.reshape((env_size, env_size))