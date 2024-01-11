import numpy as np

# The end-state is always at [env_size-1,env_size-1].
def reward_function(s, env_size):
    r = 0.0
    if (s == np.array([env_size-1, env_size-1])).all(): # If state == end state, reward = 1
        r = 1

    return r

# do not modify this function
def reward_probabilities(env_size):
    rewards = np.zeros((env_size*env_size))
    i = 0
    for r in range(env_size):
        for c in range(env_size):
            state = np.array([r,c], dtype=np.uint8)
            rewards[i] = reward_function(state, env_size)
            i+=1

    return rewards

# Check feasibility of the new state.
# If it is a possible state return s_prime, otherwise return s
def check_feasibility(s_prime, s, env_size, obstacles):
    # Outside boundaries
    if ((s_prime < 0).any()):
        return s
    if s_prime[0] >= env_size:
        return s
    if s_prime[1] >= env_size:
        return s

    # Obstacles
    #if obstacles[s_prime[0], s_prime[1]] == 1:
    #    return s

    return s_prime

def transition_probabilities(env, s, a, env_size, directions, obstacles):
    prob_next_state = np.zeros((env_size, env_size))

    # Fill in the cells corresponding to the next possible states with the probability of visiting each of them
    # Remember to check the feasibility of each new state!
    s_prime = check_feasibility(s + directions[a, :], s, env_size, obstacles)
    prob_next_state[s_prime[0], s_prime[1]] += 1/3

    s_prime = check_feasibility(s + directions[(a+1) % 4, :], s, env_size, obstacles)
    prob_next_state[s_prime[0], s_prime[1]] += 1/3

    s_prime = check_feasibility(s + directions[(a-1) % 4, :], s, env_size, obstacles)
    prob_next_state[s_prime[0], s_prime[1]] += 1/3
    
    return prob_next_state