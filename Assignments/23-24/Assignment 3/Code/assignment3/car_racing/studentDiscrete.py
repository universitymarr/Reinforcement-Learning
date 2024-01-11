import gymnasium as gym
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class ActorCritic(nn.Module):
    def __init__(self, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(ActorCritic, self).__init__()
        self.device = device

        self.gs = transforms.Grayscale()

        self.actor = nn.Sequential(
            self.layer_init(nn.Conv2d(1, 8, kernel_size = 7, stride = 4, padding = 0)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding = 2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 2)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(32 * 24 * 24, 256)),
            nn.ReLU(),
            self.layer_init(nn.Linear(256, 5))
        )

        self.critic = nn.Sequential(
            self.layer_init(nn.Conv2d(1, 8, kernel_size = 7, stride = 4, padding = 0)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding = 2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 2)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(32 * 24 * 24, 256)),
            nn.ReLU(),
            self.layer_init(nn.Linear(256, 1))
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def preProcess(self, x):
        if(len(x.shape) == 4):
            x = x.permute(0, 3, 1, 2) # n_envs, 1, 96, 96
        else:
            x = x.unsqueeze(1)        # 96, n_envs, 96, 3
            x = x.permute(1, 3, 0, 2) # n_envs, 3, 96, 96

        x = x / 255.0           # Normalize
        x = x[:, :, :83, :83]   # Crop the image to 83x83
        x = self.gs(x)          # Convert to grayscale
        return x

    def get_value(self, x):
        # Preprocess
        x = self.preProcess(x)

        # Critic
        x = self.critic(x)
        return x

    def get_action_and_value(self, x, action=None):
        # Preprocess
        x = self.preProcess(x)

        # Critic
        value = self.critic(x)

        # Actor
        logits = self.actor(x)

        logits = F.softmax(logits, dim=0)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value

class Policy(nn.Module):
    def __init__(self, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        super(Policy, self).__init__()

        # Environment
        self.device = device
        self.n_envs = 4
        self.render_mode = 'rgb_array'
        self.continuous = False
        self.initial_obs = []
        self.envs = gym.vector.SyncVectorEnv(
            lambda: self.createEnv(i+1) for i in range(self.n_envs)
        )

        # Training parameters
        self.total_timesteps = 8.0e6
        self.num_steps = 500
        self.num_minibatches = 4
        self.update_epochs = 4
        self.batch_size = int(self.n_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_updates = int(self.total_timesteps // self.batch_size)

        # PPO Hyperparameters
        self.agent = ActorCritic(self.device).to(self.device)
        self.learning_rate = 2.5e-4
        self.gamma = 0.99
        self.clip_coef = 0.2
        self.gae_lambda = 0.95
        self.ent_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        self.gae = True
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

    def createEnv(self, num_environment):
        env = gym.make('CarRacing-v2', max_episode_steps=1000, continuous=self.continuous, render_mode=self.render_mode)
        state, _ = env.reset(seed = int(42 * num_environment * 2.38))

        for _ in range(60):
            state, _, _, _, _ = env.step(0)

        self.initial_obs.append(state)

        return env
    
    def act(self, state):
        self.agent.eval()

        with torch.no_grad():
            state = torch.Tensor(state).to(self.device)
            action, _, _, _ = self.agent.get_action_and_value(state)
            return action[0].cpu().numpy()

    def train(self):
        # Set D = (o, a, log π(a|o), r, d, v) as tuple of 2D arrays
        obs = torch.zeros((self.num_steps, self.n_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.num_steps, self.n_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.n_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.n_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.n_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.n_envs)).to(self.device)

        # Initialize some variables
        global_step = 0
        next_done = torch.zeros(self.n_envs).to(self.device)

        # Initialize next observation onext = E.reset()
        next_obs = torch.Tensor(torch.from_numpy(np.asarray(self.initial_obs))).to(self.device)

        # For i = 0, 1, ..., I
        for update in tqdm(range(1, self.num_updates + 1)):
            # Annealing the rate
            frac = (1.0 - (update - 1.0)) / self.num_updates
            self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Rollout phase
            # for t = 0,1,2,..., M do
            for step in range(0, self.num_steps):
                global_step += 1 * self.n_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Get action and value from the agent and store them
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Execute the game and store the transition
                next_obs, reward, done, truncated, _ = self.envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device) # 14: Cache ot = onext and dt = dnext

            # Estimate / Bootstrap next value vnext = v(onext)
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)

                # Let advantage A = GAE(D.r, D.v, D.d, vnext, dnext, λ)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                # Let TD(λ) return R = A + D.v
                returns = advantages + values

            # Prepare the batch B = D, A, R and flatten B
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            # for epoch = 0,1,2,..., K do
            b_inds = np.arange(self.batch_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                # for mini-batch M of size m in B do
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    # Normalize advantage M.A
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Let ratio r = e^log π(M.a|M.o) −M. log π(a|o)
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                    # ================ Policy loss ================
                    ratio = torch.exp(newlogprob - b_logprobs[mb_inds])

                    clipped_ratio = ratio.clamp(min=1.0 - self.clip_coef,
                                                max=1.0 + self.clip_coef)
                    policy_reward = torch.min(ratio * mb_advantages,
                                            clipped_ratio * mb_advantages)
                    
                    pg_loss = -policy_reward.mean()

                    # ================ Value loss ================
                    # Let LV = clipped MSE(M.R, v(M.o))
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    # ================ Entropy loss ================
                    # Let LS = S[π(M.o)]
                    entropy_loss = entropy.mean()

                    # Back-propagate loss L = −Lπ + c1 * LV − c2 * LS
                    loss = pg_loss + v_loss * self.value_loss_coef - self.ent_coef * entropy_loss 

                    self.optimizer.zero_grad()
                    loss.backward()

                    # Clip maximum gradient norm of θπ and θv to 0.5
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)

                    # Step the optimizer O to initiate gradient descent
                    self.optimizer.step()

            # Calculate the mean for each environment (mean along axis 0)
            # and the overall mean (mean across all environments, mean along axis 1)
            mean_per_environment = torch.mean(rewards, dim=0)
            mean_across_all_environments = torch.mean(rewards, dim=1)

            # Print the results
            print(F"\nIteration: {update} | Timestep: {global_step}\n\
                    - Total Loss: {loss.item()}\n\
                    - Policy Loss: {pg_loss.item()}\n\
                    - Value Loss: {v_loss.item()}\n\
                    - Entropy Loss: {entropy_loss.item()}\n\
                    - Mean per environment: {mean_per_environment}\n\
                    - Mean across all environments: {mean_across_all_environments.mean()}\n")
            
            # Save the model every 10 iterations
            if(update % 10 == 0):
                self.save(iteration=update)

    def save(self, iteration=None):
        torch.save(self.state_dict(), f'model_{iteration}.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    