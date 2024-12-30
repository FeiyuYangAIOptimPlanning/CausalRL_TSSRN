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
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork

from common.utils import masked_entropy, index_to_one_hot, to_tensor_var, npABCDToTensorList, action2code, vae_loss_function,binary_to_one_hot,addScalarFcn


class MAA2C(Agent):
    """

    """
    def __init__(self, 
                 env,
                 n_agents,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=256,
                 reward_gamma=0.92, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=256,
                 actor_output_act=nn.functional.log_softmax, critic_loss="huber",
                 actor_lr=0.001, critic_lr=0.0002,
                 optimizer_type="adam", entropy_reg=0.0005,
                 max_grad_norm=10, batch_size=256, episodes_before_train=100,
                 epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=3500,
                 use_cuda=True, training_strategy="centralized",
                 actor_parameter_sharing=False, critic_parameter_sharing=True):
        super(MAA2C, self).__init__(env,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        assert training_strategy in ["cocurrent", "centralized"]

        self.state_dims=[58,58,58,82,82,82,82,58,58,58,74,58,58,58,58,58,58,74]
        self.total_action_dims=[7, 7, 7, 10, 10, 10, 10, 7, 7, 7, 9, 7, 7, 7, 7, 7, 7, 9]

        self.n_agents = n_agents
        self.roll_out_n_steps = roll_out_n_steps
        self.training_strategy = training_strategy
        self.actor_parameter_sharing = actor_parameter_sharing
        self.critic_parameter_sharing = critic_parameter_sharing

        self.ep_rewards=[[] for _ in range(self.n_agents)]
    
        device=None
        z_dim=32
        if use_cuda:
            device='cuda'
        else:
            device='cpu'
        self.actors = [ActorNetwork(state_dim+z_dim, self.actor_hidden_size, action_dim, self.actor_output_act,index=idx,device=device,z_dim=z_dim) for idx,(state_dim,action_dim) in enumerate(zip(self.state_dims,self.total_action_dims))] #Edited
        
        if self.training_strategy == "cocurrent":
            
            pass
        elif self.training_strategy == "centralized":
            critic_state_dim = sum(self.state_dims)
            critic_action_dim = 4*18
            critic_z_dim=z_dim*18
            self.critics = [CriticNetwork(critic_state_dim+critic_z_dim, critic_action_dim, 256, 1)] * self.n_agents
        if optimizer_type == "adam":
            self.actor_optimizers = [Adam([
                {'params':a.causal_module.parameters(),'lr':self.actor_lr},
                {'params':a.fc1.parameters(),'lr':self.actor_lr/2,'weight_decay': 1e-6},
                {'params':a.fc2.parameters(),'lr':self.actor_lr/2,'weight_decay': 1e-6},
                ]) for a in self.actors]
            
            self.critic_optimizers = [Adam(c.parameters(), lr=self.critic_lr, weight_decay=1e-6) for c in self.critics]
      
            for param_group in self.actor_optimizers[0].param_groups:
                for param in param_group['params']:
                    print(param.shape)  
          
            param_to_name = {id(param): name for name, param in self.actors[0].named_parameters()}

            for param_group in self.actor_optimizers[0].param_groups:
                for param in param_group['params']:
                    param_id = id(param)
                    if param_id in param_to_name:
                        print(f"层名称: {param_to_name[param_id]}, 形状: {param.shape}")

            for param_group in self.critic_optimizers[0].param_groups:
                for param in param_group['params']:
                    print(param.shape)  

            param_to_name = {id(param): name for name, param in self.critics[0].named_parameters()}

            for param_group in self.critic_optimizers[0].param_groups:
                for param in param_group['params']:
                    param_id = id(param)
                    if param_id in param_to_name:
                        print(f"层名称: {param_to_name[param_id]}, 形状: {param.shape}")

        elif optimizer_type == "rmsprop":
            self.actor_optimizers = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critic_optimizers = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        if self.actor_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.actors[agent_id] = self.actors[0]
                self.actor_optimizers[agent_id] = self.actor_optimizers[0]
        
        if self.critic_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.critics[agent_id] = self.critics[0]
                self.critic_optimizers[agent_id] = self.critic_optimizers[0]
        
        if self.use_cuda:
            for a in self.actors:
                a.cuda()
            for c in self.critics:
                c.cuda()

    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
     
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action,_,_,_,_ = self.exploration_action(self.env_state)
            next_state, reward, done, _,_ = self.env.step(action)
            for idx,r in enumerate(reward):
                self.ep_rewards[idx].append(r.item())
          
            done_total=max(done)
            actions.append([action2code(a) for a in action])
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done_total:
                self.env_state = self.env.reset()
                break
  
        if done_total:
            final_r = [0.0] * self.n_agents
           
            self.n_episodes += 1
            self.episode_done = True
            for idx in range(self.n_agents):
                addScalarFcn(f'reward/agent{idx}',sum(self.ep_rewards[idx]),self.n_episodes)
            addScalarFcn(f'reward/episode',sum([sum(self.ep_rewards[idx]) for idx in range(self.n_agents)]),self.n_episodes)
            for idx in range(self.n_agents):
                self.ep_rewards[idx].clear()
        else:
            self.episode_done = False
            final_action,z_st,_,_,_ = self.action(final_state)
            binary_action = [action2code(a,True,self.use_cuda) for a in final_action]
            action_var=to_tensor_var(binary_action, self.use_cuda).view(-1, self.n_agents*4)
            final_r = self.value(th.concat(z_st,dim=-1), action_var)
      
        rewards = np.array(rewards)
        for agent_id in range(self.n_agents):
            rewards[:,agent_id] = self._discount_reward(rewards[:,agent_id], final_r[agent_id])
        rewards = rewards.tolist()
        self.n_steps += 1
        self.memory.push(states, actions, rewards)

    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass
        else:
            batch = self.memory.sample(self.batch_size)
            states_var = npABCDToTensorList(batch.states, self.use_cuda)
            actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, 4)
            rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
            
            z_st_total=[]
            for idx in range(self.n_agents):
                _, z_st_tmp,_,_,_,_= self.actors[idx](states_var)
                z_st_total.append(z_st_tmp)
                
            whole_critic_states_var = th.concat(z_st_total,dim=-1)
            whole_actions_var = actions_var.view(-1, 18 * 4)
            whole_critic_states_var = th.concat(z_st_total,dim=-1)
            
            critic_mean_loss=0
            for agent_id in range(self.n_agents):
            
                self.actor_optimizers[agent_id].zero_grad()
                self.critic_optimizers[agent_id].zero_grad() 

                values = self.critics[agent_id](whole_critic_states_var.detach(), whole_actions_var)

                action_log_probs, z_st_this,vaelossinput,predilossinput,predirewardlossinput,mask= self.actors[agent_id](states_var)             
                valid_entropy_batch_mean=masked_entropy(action_log_probs,mask.detach())
                entropy_loss = -valid_entropy_batch_mean
                action_onehot=binary_to_one_hot(actions_var[:,agent_id,:],self.total_action_dims[agent_id],self.use_cuda)
                action_log_probs = th.sum(action_log_probs * action_onehot, -1)
                advantages = rewards_var[:,agent_id,:] - values.detach()
                pg_loss = -th.mean(action_log_probs * advantages.squeeze())
                vae_loss=vae_loss_function(*vaelossinput)
                r_pred_loss=nn.functional.smooth_l1_loss(*predirewardlossinput)+nn.functional.mse_loss(*predirewardlossinput)
                predict_loss=nn.functional.mse_loss(*predilossinput)+r_pred_loss
               
                actor_loss = pg_loss*100 +  entropy_loss*(0.05/entropy_loss.item())+ vae_loss*0.1 + predict_loss*0.01 
                
                addScalarFcn(f'entropy_loss/agent_{agent_id}_entropy_loss', 1* entropy_loss.item(),(self.n_steps-1))
                addScalarFcn(f'advantages/agent_{agent_id}_advantages', th.mean(advantages).item(),(self.n_steps-1))
                addScalarFcn(f'pg_objective/agent_{agent_id}_pg_objective', pg_loss.item(),(self.n_steps-1))
                addScalarFcn(f'vae_loss/agent_{agent_id}_vae_loss', vae_loss.item(),(self.n_steps-1))
                addScalarFcn(f'predict_loss/agent_{agent_id}_predict_loss', predict_loss.item(),(self.n_steps-1))
                addScalarFcn(f'reward_predict_loss/agent_{agent_id}_reward_predict_loss', r_pred_loss.item(),(self.n_steps-1))
                addScalarFcn(f'actorloss/actor_{agent_id}_loss', actor_loss.item(),(self.n_steps-1))
                
                
                actor_loss.backward()
                self.actor_optimizers[agent_id].step()
                
                target_values = rewards_var[:,agent_id,:]
                if self.critic_loss == "huber":
                    critic_loss = nn.functional.smooth_l1_loss(values, target_values)
                else:
                    critic_loss = nn.MSELoss()(values, target_values)
                critic_mean_loss+=critic_loss.item()
                critic_loss.backward()

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
                
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), self.max_grad_norm)
                
                
                self.critic_optimizers[agent_id].step()
            addScalarFcn('central_critic_loss',critic_mean_loss/18,(self.n_steps-1)) 
           
    def _softmax_action(self, state):
        state_var = npABCDToTensorList(state, self.use_cuda)#Edited
        softmax_action = [None for _ in range(self.n_agents)]
        z_st_agent_list=[None for _ in range(self.n_agents)]
        vaelossinput_agent_list=[None for _ in range(self.n_agents)]
        predilossinput_agent_list=[None for _ in range(self.n_agents)]
        for agent_id in range(self.n_agents):
            out,z_st,vaelossinput,predilossinput,predirewardlossinput,mask=self.actors[agent_id](state_var)#policy network forward
            softmax_action_var = th.exp(out)
            if self.use_cuda:
                softmax_action[agent_id] = softmax_action_var.data.cpu().numpy()[0]
            else:
                softmax_action[agent_id] = softmax_action_var.data.numpy()[0]
            z_st_agent_list[agent_id]=z_st
            vaelossinput_agent_list[agent_id]=vaelossinput
            predilossinput_agent_list[agent_id]=predilossinput
        return softmax_action,z_st_agent_list,vaelossinput_agent_list,predilossinput_agent_list,predirewardlossinput

    def exploration_action(self, state):
        softmax_action,z_st,vaelossinput,predilossinput,predirewardlossinput = self._softmax_action(state)
        actions = [0]*self.n_agents
        for agent_id in range(self.n_agents):
            actions[agent_id] = np.random.choice(self.total_action_dims[agent_id],p=softmax_action[agent_id])
        return actions,z_st,vaelossinput,predilossinput,predirewardlossinput

 
    def action(self, state, evaMode=False):
        softmax_actions,z_st,_,_, _ = self._softmax_action(state)
        actions = [0]*self.n_agents
        for agent_id in range(self.n_agents):
            if evaMode==False:
                actions[agent_id] = np.random.choice(self.total_action_dims[agent_id],p=softmax_actions[agent_id])
            else:
                actions = [np.argmax(softmax_action,axis=-1) for softmax_action in softmax_actions]
        return actions,z_st,_,_,_

   
    def value(self, state, action):
        '''
       
        '''
       
        whole_state_var = state
        whole_action_var = action
        values = [0]*self.n_agents
        
        value_var = self.critics[0](whole_state_var.detach(), whole_action_var.detach())
        for agent_id in range(self.n_agents):
            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

