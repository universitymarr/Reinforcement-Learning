import json

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm

# Gaussian distribution for the agent
class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()

    # Returns mean, actions, log_actions, entropy starting from mean and std
    def forward(self, mean_actions, std_actions, old_actions):
        distribution = Normal(mean_actions, std_actions)
        actions_with_exploration = distribution.sample()

        if old_actions is None:
            log_actions = distribution.log_prob(actions_with_exploration)
        else:
            log_actions = distribution.log_prob(old_actions)

        return distribution.mean, actions_with_exploration, log_actions, distribution.entropy()
    
# PPO model for the agent (actor-critic)
class PPO(nn.Module):
    def __init__(self, output_size):
        super(PPO, self).__init__()

        # Convolutional layers
        self.conv2d_0 = nn.Conv2d(1, 8, kernel_size=4, stride=2)
        self.conv2d_1 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv2d_2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv2d_3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv2d_4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2d_5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)

        # Actor
        self.action_mean = nn.Linear(256, output_size)
        self.action_std = nn.Linear(256, output_size)

        # Critic
        self.critic_output = nn.Linear(256, 1)

        # The SoftPlus is a smooth continuous version of reluLayer.
        # This layer is useful for creating continuous Gaussian policy deep neural networks,
        # for which the standard deviation output must be positive.
        self.std_activation = nn.Softplus()
        self.relu = nn.ReLU()
        self.gaussian = Gaussian()
        
        # Initialize weights with Orthogonal initialization
        for layer in [self.conv2d_0 , self.conv2d_1, self.conv2d_2, self.conv2d_3, self.conv2d_4, self.conv2d_5, self.action_mean, self.action_std,\
                      self.critic_output]:
            torch.nn.init.orthogonal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, old_actions=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convolutional layers
        x = self.relu(self.conv2d_0(x))
        x = self.relu(self.conv2d_1(x))
        x = self.relu(self.conv2d_2(x))
        x = self.relu(self.conv2d_3(x))
        x = self.relu(self.conv2d_4(x))
        x = self.relu(self.conv2d_5(x))

        x = x.view(x.shape[0], -1)  # Reshape to (kernel_size, 256)

        # ===== Actor =====
        x_action_mean = (self.action_mean(x))
        x_action_std = self.std_activation(self.action_std(x))

        mean, actions, log_actions, entropy = self.gaussian(x_action_mean, x_action_std, old_actions)

        # ===== Critic =====
        x_value = self.critic_output(x)

        return mean, actions, log_actions, entropy, x_value

class Policy(nn.Module):
    def __init__(self, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(Policy, self).__init__()
        # CUDA
        self.device = device

        # Environment
        self.continuous = True
        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode="rgb_array")

        # Hyperparameters for PPO
        self.gamma = 0.98
        self.clip_eps = 0.2
        self.value_factor = 0.5
        self.entropy_factor = 0.005

        # Training parameters
        self.n_episodes = 10000
        self.n_updates_per_episode = 5

        # Create the agent
        self.agent = PPO(output_size=3).to(device)

    # Compute losses for training phase (policy loss, value loss and entropy loss)
    def compute_losses(self, states, actions, returns, log_actions, advantages) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute new log actions
        _, _, log_actions_new_policy, entropy, values = self.agent(states, actions)

        # 27. Compute ratios
        ratios = torch.exp(log_actions_new_policy - log_actions)

        # 28. Compute policy loss
        policy_loss = torch.min(ratios*advantages, torch.clip(ratios, 1 - self.clip_eps, 1 + self.clip_eps)*advantages)
        policy_loss = -torch.mean(policy_loss)

        # 29. Compute value loss // MSE
        value_loss = (advantages + values**2).mean()
        
        # 30. Compute entropy loss
        entropy_loss = -torch.mean(entropy)

        return policy_loss, value_loss, entropy_loss

    # Rollout for training phase
    def rollout(self, it):  

        # 5: Initialize next observation onext = E.reset()
        state, _ = self.env.reset()

        # 6: Initialize next done flag dnext = [0, 0, ..., 0] # length N. In this case N = 1
        done = False

        # 10: Set D = (o, a, log π(a|o), r, d, v) as tuple of 2D arrays
        memory = []

        streak = 0
        total_reward = 0

        # Stack various experiences in memory
        print(F"\nGetting rollout for episode n.{it}..")

        # 8. for t = 0,1,2,..., M do. In this case M = 1000
        while not done:
            _, action, log_action = self.forward(state)

            fixed_action = action.copy()

            next_state, reward, done, _, _ = self.env.step(fixed_action)
            total_reward += reward

            if total_reward > 900:
                reward = 100
                while not done:
                    _, _, done, _ = self.env.step(fixed_action)
            else:
                if reward < 0:
                    streak += 1
                    if streak > 100:
                        reward = -100
                        while not done:
                            _, _, terminated, truncated, _ = self.env.step(fixed_action)
                            done = terminated or truncated
                else:
                    streak = 0
            # Store the experience in memory
            memory.append([state, action, reward, log_action])
            # Cache ot = onext
            state = next_state

        states, actions, rewards, log_actions = map(np.array, zip(*memory))

        # Compute discounted rewards (returns) in reverse order
        discount = 0
        discountedRewards = np.zeros((len(rewards)))

        for i in reversed(range(len(rewards))):
            discount = rewards[i] + discount * self.gamma
            discountedRewards[i] = discount

        # States, actions, returns and log_actions
        return self.to_torch(states).mean(dim=3).unsqueeze(dim=1), \
                self.to_torch(actions), \
                self.to_torch(discountedRewards).reshape(-1,1), \
                self.to_torch(log_actions), \
                total_reward

    # Rollout forward
    def forward(self, x: np.ndarray, stabilize=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Scale the state between 0 and 1
        # Also, get the tensor version, reshape it and turn it into B&W
        x = x / 255.0
        x = self.to_torch(x).mean(dim=2).reshape(1, 1, x.shape[0], x.shape[1])

        # Get the mean, actions and log_actions
        mean, actions, log_actions, _, _ = self.agent(x)

        actions = actions[0]

        if(stabilize):
            actions[0] = torch.clamp(actions[0], min=-1, max=1)
            actions[1] = torch.clamp(actions[1], min=0, max=0.5)
            actions[2] = torch.clamp(actions[2], min=0, max=1)

        actions = actions.detach().cpu().numpy()
        log_actions = log_actions[0].detach().cpu().numpy()
        mean = mean[0].detach().cpu().numpy()

        return mean, actions, log_actions

    # Act for evaluation phase
    # Stabilize is used to avoid the car to drive very fast and slip
    def act(self, x: np.ndarray, stabilize=True):

        # Scale the state between 0 and 1
        # Also, get the tensor version, reshape it and turn it into B&W
        x = x / 255.0
        x = self.to_torch(x).mean(dim=2).reshape(1, 1, x.shape[0], x.shape[1])

        _, actions, _, _, _ = self.agent(x)
        actions = actions[0]

        if(stabilize):
            actions[0] = torch.clamp(actions[0], min=-1, max=1)
            actions[1] = torch.clamp(actions[1], min=0, max=0.5)
            actions[2] = torch.clamp(actions[2], min=0, max=1)

        return actions.detach().cpu().numpy()

    def train(self):
        # 4. Initialize Adam optimizer O
        agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr = 0.00015)

        scores = []

        # 8: for i = 0,1,2,..., I do
        with tqdm(total=self.n_episodes, desc="Episode") as pbar:
            iteration = 0

            while iteration < self.n_episodes:
                with torch.no_grad():
                    self.agent.eval()
                    states, actions, returns, log_actions, episode_score = self.rollout(it=iteration)

                scores.append(episode_score)
                print(f"Current score: {scores[-1]}")
                print(f"episode: {len(scores)}")

                # 20. Estimate / Bootstrap next value vnext = v(onext)
                _, _, _, _, values = self.agent(states)

                # Since we calculated the returns, the pseudo-code says: R = A + D.v
                # In this case, we calculate A as R - V and normalize them
                advantages = returns - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Update the agent
                self.agent.train()

                # 24. for epoch = 0,1,2,..., K do. In this case K = 5
                for n_upd in range(self.n_updates_per_episode):
                    agent_optimizer.zero_grad()
                    policy_loss, value_loss, entropy_loss = self.compute_losses(states, actions, returns, log_actions, advantages)

                    # Calculate the whole loss
                    loss = 2 * policy_loss + self.value_factor * value_loss + self.entropy_factor * entropy_loss
                    print(f'Loss at step {n_upd}: {loss:.5f}')

                    # Backpropagate the loss
                    loss.backward()

                    # 32. Clip maximum gradient norm of θπ and θv to 0.5
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)

                    # 33. Step the optimizer O to initiate gradient descent
                    agent_optimizer.step()

                pbar.update(1)
                iteration += 1

                # Save the model if the score is the best
                if(episode_score >= max(scores)):
                    self.save(episode_score, iteration)
                    torch.save(self.agent, f'agent_checkpoint_{iteration}.pt')

                # Save the results every 10 episodes
                if iteration % 10 == 0:
                    with open(f'results_checkpoint_{iteration}.json', 'w') as f:
                        json.dump({"scores": scores}, f)

        with open('results.json', 'w') as f:
            json.dump({"scores": scores}, f)    # Dump all the scores in a single big file

        torch.save(self.agent, 'agent.pt')      # Save the agent after the whole training process
        return

    def save(self, episode_score=None, iteration=None):
        torch.save(self.state_dict(), f'model_{iteration}_{episode_score:.5f}.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to_torch(self, tensor):
        return torch.tensor(tensor.copy(), dtype=torch.float32, device=self.device)