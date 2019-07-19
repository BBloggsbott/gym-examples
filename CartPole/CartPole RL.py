#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import warnings
warnings.simplefilter('ignore')


# In[2]:


gamma = 0.99
seed = 1234


# In[3]:


env = gym.make('CartPole-v1')
env.seed(seed)
torch.manual_seed(seed)


# In[4]:


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128,2)
        
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


# In[5]:


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr = 1e-2)
eps = np.finfo(np.float32).eps.item()


# In[6]:


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


# In[7]:


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob*R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# In[9]:


running_reward = 10
prev_running_reward = 0
saved_at = 10
for i_episode in count(1):
    state, ep_reward = env.reset(), 0
    for t in range(1, 1000):
        action = select_action(state)
        state, reward, done, _ = env.step(action)
        env.render()
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break
    env.close()
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    finish_episode()
    if i_episode % 10 == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward), end=' ')
        if running_reward > prev_running_reward or i_episode == 10:
            torch.save(policy, "cartpole_policy.pth")
            saved_at = i_episode
            print('Saved Model.')
        elif i_episode - saved_at > 30:
            print('Average Reward not increased for {} episodes. Quitting.'.format(i_episode - saved_at))
            break
        else:
            print()
        prev_running_reward = running_reward
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, t))
        break

