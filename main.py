#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:23:48 2017

@author: tsu
"""
import numpy as np
import gym
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

LAMBDA = 0.8
EPISODES = 1000
STD_POLICY = 0.2

class Agent(object):
    
    def __init__(self):
        self.policy = _Policy()
        self.critic = _Critic()
        self.policy_optim = optim.Adam(self.policy.parameters())
        self.critic_optim = optim.Adam(self.critic.parameters())
        
    def reset_obs(self, observation):
        self.obs = Variable(torch.Tensor(observation), requires_grad=False)
            
    def act(self):
        self.action = torch.normal(self.policy(self.obs), STD_POLICY) 
        return self.action.data.numpy()
    
    def sense(self, observation, reward):
        self.next_obs = Variable(torch.Tensor(observation), requires_grad=False)
        self.reward = Variable(torch.Tensor([reward]), requires_grad=False)
    
    def learn(self):
        delta = self.reward + LAMBDA * self.critic(self.next_obs) \
                - self.critic(self.obs)
                
        # training for critic
        self.critic.zero_grad()
        V_now = self.critic(self.obs)
        V_now.backward()
        for grad in self.critic.parameters():
            grad = -grad * delta
        self.critic_optim.step()
        
        # training for policy
        self.policy.zero_grad()
        temp = self.action.detach()-self.policy(self.obs)
        loss = -torch.dot(temp,temp)
        loss.backward()
        for grad in self.policy.parameters():
            grad = grad * delta
        self.policy_optim.step()
    
    
class _Policy(nn.Module):
    def __init__(self):
        super(_Policy, self).__init__()
        self.fc1 = nn.Linear(24,12)
        self.fc2 = nn.Linear(12,4)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x
        
class _Critic(nn.Module):
    def __init__(self):
        super(_Critic, self).__init__()
        self.fc1 = nn.Linear(24,12)
        self.fc2 = nn.Linear(12,1)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x
    
    
if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')
    agent = Agent()
    for i_episode in range(EPISODES):
        initial_obs = env.reset()
        agent.reset_obs(initial_obs)
        for t in range(300):
            env.render()
            action = agent.act()
            print(action)
            observation, reward, done, _ = env.step(action)
            agent.sense(observation, reward)
            agent.learn()
            if done:
                break
            
            
        
    
    
    
    
    
    