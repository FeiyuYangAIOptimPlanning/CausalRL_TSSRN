# MIT License

# Copyright (c) 2024 Feiyu Yang 杨飞宇

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# This code is based on the project pytorch-DRL by Chenglong Chen (2017).
# Modifications by Feiyu Yang (2024).
# 
# Licensed under the MIT License.

# MIT License

# Copyright (c) 2017 Chenglong Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import torch as th

import numpy as np

from common.Memory import ReplayMemory
from common.utils import identity


class Agent(object):
    """
    A unified agent interface:
    - interact: interact with the environment to collect experience
        - _take_one_step: take one step
        - _take_n_steps: take n steps
        - _discount_reward: discount roll out rewards
    - train: train on a sample batch
        - _soft_update_target: soft update the target network
    - exploration_action: choose an action based on state with random noise
                            added for exploration in training
    - action: choose an action based on state for execution
    - value: evaluate value for a state-action pair
    - evaluation: evaluation a learned agent
    """
    def __init__(self, env,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.92, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=256,
                 actor_output_act=identity, critic_loss="huber",
                 actor_lr=0.001, critic_lr=0.0002,
                 optimizer_type="adam", entropy_reg=0.0005,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=3500,
                 use_cuda=True):
        self.env = env
        self.state_dim = None
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = 1

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = ReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.target_tau = 0.01

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()

    # agent interact with the environment to collect experience
    def interact(self):
        pass

    # # take one step
    # def _take_one_step(self):
    #     if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
    #         self.env_state = self.env.reset()
    #         self.n_steps = 0
    #     state = self.env_state
    #     action,_,_,_,_ = self.exploration_action(self.env_state)
    #     next_state, reward, done, _,_ = self.env.step(action)
    #     if done:
    #         if self.done_penalty is not None:
    #             reward = reward            #Edited 不需要专门的奖励？
    #         #next_state = [0] * len(state) #Edited 不需要专门的状态重置？
    #         self.env_state = self.env.reset()
    #         self.n_episodes += 1
    #         self.episode_done = True
    #     else:
    #         self.env_state = next_state
    #         self.episode_done = False
    #     self.n_steps += 1
    #     self.memory.push(state, action, reward, next_state, done)

    # # take n steps
    # def _take_n_steps(self):
    #     if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
    #         self.env_state = self.env.reset()
    #         self.n_steps = 0
    #     states = []
    #     actions = []
    #     rewards = []
    #     # take n steps
    #     for i in range(self.roll_out_n_steps):
    #         states.append(self.env_state)
    #         action,_,_,_,_  = self.exploration_action(self.env_state)
    #         next_state, reward, done, _,_ = self.env.step(action)
    #         actions.append(action)
    #         if done and self.done_penalty is not None:
    #             reward = reward            #Edited 不需要专门的奖励？
    #         rewards.append(reward)
    #         final_state = next_state
    #         self.env_state = next_state
    #         if done:
    #             self.env_state = self.env.reset()
    #             break
    #     # discount reward
    #     if done:
    #         final_value = 0.0
    #         self.n_episodes += 1
    #         self.episode_done = True
    #     else:
    #         self.episode_done = False
    #         final_action,_,_,_,_  = self.action(final_state)#这里直接取action
    #         final_value = self.value(final_state, final_action)
    #     rewards = self._discount_reward(rewards, final_value)
    #     self.n_steps += 1
    #     self.memory.push(states, actions, rewards)

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    # train on a sample batch
    def train(self):
        pass

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state):
        pass

    # choose an action based on state for execution
    def action(self, state):
        pass

    # evaluate value for a state-action pair
    def value(self, state, action):
        pass

    # evaluation the learned agent
    def evaluation(self, env, eval_episodes=10):
        rewards = []
        infos = []
        for i in range(eval_episodes):
            rewards_i = []
            infos_i = []
            state = env.reset()
            action,_,_,_,_ = self.action(state,evaMode=True)
            state, reward, dones,_, info = env.step(action)
            done = max(dones)
            rewards_i.append(reward)
            infos_i.append(info)
            while not done:
                action,_,_,_,_ = self.action(state,evaMode=True)
                state, reward, dones,_, info = env.step(action)
                done = max(dones)
                rewards_i.append(reward)
                infos_i.append(info)
            rewards.append(rewards_i)
            infos.append(infos_i)
        return rewards, infos#rewards->list_ep(list_step->(array(18)))