import random
import numpy as np
# import gymnasium as gym
# import time
# from gymnasium import spaces
# import os
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle

class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:
    def __init__(self, env):
        self.env = env
        self.RBF_components = 200 # Number of RBF Kernels

        # To accumulate experiences
        self.observation_examples = np.array([env.observation_space.sample() for x in range(20000)])

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.encoder = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=0.5, n_components=100))
        ])

        # Fit both scaler and encoder on previous experiences
        print("-----------------------------------------------")
        print("Fitting the Encoder to previous experiences...")
        self.scaler.fit(self.observation_examples)
        self.encoder.fit(self.scaler.transform(self.observation_examples))
        print("Done.")
        print("-----------------------------------------------")

    def encode(self, state): # modify
        features = self.scaler.transform([state])
        features = self.encoder.transform(features)
        return features.reshape(-1)

    @property
    def size(self): # modify
        return self.RBF_components

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder, alpha=0.01, alpha_decay=1, 
                 gamma=0.9999, epsilon=0.3, epsilon_decay=0.9, final_epsilon=0.2, lambda_=0.6): # modify if you want (e.g. for forward view)
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size) # -> Shape (3, 200)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.lambda_ = lambda_
        
    def Q(self, feats):
        feats = feats.reshape(-1, 1)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)

        # ========================================================
        # Update weights (Pag. 29 of PDF 8 - Linear Approximation)
        # 
        # δ_t = Rt+1 + γ * Q(s_t+1,a_t+1,w) - Q(s_t,a_t,w)
        # e_t = γ * λ * e_t-1 + ∇_w Q(s_t,a_t,w)
        # w_t+1 = w_t - α * δ_t * e_t
        #
        # ========================================================
        delta = reward + (1 - done) * self.gamma * self.Q(s_prime_feats).max() - self.Q(s_feats)[action]    # δ_t
        self.traces = self.gamma * self.lambda_ * self.traces                                               # e_t-1
        self.traces[action] = self.traces[action] + s_feats                                                 # e_t
        self.weights[action] = self.weights[action] + self.alpha * delta * self.traces[action]              # w_t
        
    def update_alpha_epsilon(self): # do not touch
        self.epsilon = max(self.final_epsilon, self.epsilon*self.epsilon_decay)
        self.alpha = self.alpha*self.alpha_decay
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None): # do not touch
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
       
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        print(f'ep | eval | epsilon | alpha')
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)
                
                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
