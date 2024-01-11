import numpy as np
import random
from statistics import mean
from matplotlib import pyplot as plt

from tqdm import tqdm

def evaluateQTable(env, q_table, n_episodes=10):
    cum_rews = []                                                           # Accumulated rewards for episodes
    for ep in range(n_episodes):                                            # Loop throughout the episodes
        cum_rew = 0                                                         # Accumulate reward until we reach the goal 
        done = False
        state, _ = env.reset()
        while not done:
            action = np.argmax(q_table[state])                              # Take directly argmax from the Q-Table. Meaning, we exploit the learned values
            state, reward, terminated, truncated, info = env.step(action)   # Run a simulation step in the environment with the best action
            done = terminated or truncated
            cum_rew += reward                                               # Accumulate all the rewards until we reach the goal
        cum_rews.append(cum_rew)                                            # Insert the accumulated rewards for this episode.                 
    return mean(cum_rews)                                                   # Return the mean reward we got from running n episodes.

# Choose the action with epsilon-greedy strategy
def epsilon_greedy_action(env, Q, state, epsilon):
    if (random.random() < epsilon):
        action = env.action_space.sample()  # In this case, let's use the action_space.sample() from Gymnasium,
                                            # which returns a random action from the action space.
    else:
        action = np.argmax(Q[state])        # Here we'll take the argmax over Q. Greedy
    return action

def sarsa_lambda(env, alpha=0.2, gamma=0.99, lambda_= 0.5, initial_epsilon=1.0, n_episodes=10000):

    ####### Hyperparameters
    # alpha = learning rate
    # gamma = discount factor
    # lambda_ = elegibility trace decay
    # initial_epsilon = initial epsilon value
    # n_episodes = number of episodes

    # Define the Q-Table as a random distribution instead of 0
    Q = np.random.rand(env.observation_space.n, env.action_space.n)
    print("TRAINING STARTED")
    print("...")
    # init epsilon
    epsilon = initial_epsilon
    received_first_reward = False

    mean_evaluation_rewards = []
    evaluation_rewards = []

    for ep in tqdm(range(n_episodes)):

        # Initialize Eligibility traces at the start of the episode
        E = np.zeros((env.observation_space.n, env.action_space.n))

        ep_len = 0
        done = False

        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)

        while not done:
            ############## simulate the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_len += 1
            # env.render()
            
            # Random or Greedy action
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)

            # Q[state, action]           = Take the old value from Q-Table -> Q(S, A)
            # Q[next_state, next_action] = Take the new value from Q-Table -> Q(S', A')
            delta = reward + (1 - done) * gamma * Q[next_state, next_action] - Q[state, action]

            # E(S, A) = E(S, A) + 1
            E[state, action] = E[state, action] + 1

            # For all s ∈ S
                # For all a ∈ A(s)
                # Q(s, a) = Q(s, a) * α * δ * E(s, a)
                # E(s, a) = γ * λ * E(s, a) 
            Q = Q + alpha * delta * E
            E = E * gamma * lambda_
                
            if not received_first_reward and reward > 0:
                received_first_reward = True
                print(F"Received first reward at episode n. {ep}")

            # update current state
            state = next_state      # S = S'
            action = next_action    # A = A'

        evaluation_rewards.append(evaluateQTable(env, Q, n_episodes=10))        # Mean reward with this Q-Table over 10 episodes
        mean_evaluation_rewards.append(mean(evaluation_rewards[-n_episodes:])) 
        #print(f"Episode {ep} finished after {ep_len} steps.")

        # update current epsilon
        if received_first_reward:
            epsilon = 0.99 * epsilon

    plt.title("Mean Evaluation Rewards")
    plt.xlabel('Episode')
    plt.ylabel('Mean reward')
    plt.grid(True)
    plt.plot(mean_evaluation_rewards)
    plt.savefig(F"Mean Evaluation Rewards - Lambda {lambda_}.png")

    print("TRAINING FINISHED")
    return Q